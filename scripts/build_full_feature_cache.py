#!/usr/bin/env python3
"""Stage 1: Build feature cache from full pipeline (API + GPU multi-threaded).

Reference: ARCHITECTURE.md \u00a77.1, \u00a79.2

This script:
  1. Iterates all train/test samples (sample-level parallel via ThreadPoolExecutor(n))
  2. For each sample, runs the full pipeline with multi-threaded API calls:
     - Module 1: ImageBind encoding (GPU, single-threaded \u2014 \u00a79.2.6)
     - Module 2: Baidu search \u2192 HTTP fetch \u2192 Qwen extraction (ThreadPoolExecutor(7) \u2014 \u00a79.2.3)
     - Module 3: Ta/Tb style rewriting (ThreadPoolExecutor(2)) \u2192 Qwen3-32B analysis \u2014 \u00a79.2.4
  3. Saves intermediate representations to JSONL cache files

Supports resumable execution (skips already-processed IDs).
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Load .env file manually (no dotenv dependency)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("\"'"))

# Disable rate limiting per \u00a79.2.5 (all intervals set to 0)
os.environ.setdefault("BAIDU_MIN_INTERVAL_SECONDS", "0.0")
os.environ.setdefault("QWEN8B_MIN_INTERVAL_SECONDS", "0.0")
os.environ.setdefault("QWEN32B_MIN_INTERVAL_SECONDS", "0.0")

# Project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "ImageBind"))

from factyle.data.dataset import FakeSVDataset, FakeTTDataset, FakeNewsSample
from factyle.features.text import build_text_aux
from factyle.utils.config import load_config
from factyle.clients.qwen_client import QwenClient
from factyle.retrieval.entity_extractor import Module2EntityExtractor, ENTITY_TYPES
from factyle.retrieval.style_rewriter import Module3StyleRewriter
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind import data as ib_data


# =============================================================================
# Feature Cache Builder (Full Pipeline)
# =============================================================================


class FullFeatureCacheBuilder:
    """Build feature cache with full API + GPU pipeline.

    Multi-threading architecture per \u00a79.2:
      - Sample iteration: ThreadPoolExecutor(8) \u2014 sample-level parallel
      - GPU (ImageBind, BERT): serialized via threading.Lock
      - Module 2: ThreadPoolExecutor(7) \u2014 7 entity types in parallel
      - Module 3: ThreadPoolExecutor(2) \u2014 Ta/Tb rewriting in parallel
    """

    def __init__(self, config, device: str = "cuda", workers: int = 8, freeze: bool = False,
                 cache_dir: Optional[str] = None, module3_text_mode: str = "both",
                 offline_m3: bool = False):
        self.config = config
        self.device = device
        self.freeze = freeze  # \u00a77.3.5: set cache read-only after build
        self.cache_dir = Path(cache_dir or config.cache.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.frame_cache_dir = Path(config.cache.cache_dir).parent / "frame_cache"
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)

        # Sample-level parallelism (GPU-locked for safety)
        self.sample_workers = workers
        self._gpu_lock = threading.Lock()    # serializes ImageBind + BERT
        self._write_lock = threading.Lock()  # serializes JSONL + log writes
        self._stats_lock = threading.Lock()  # serializes stats updates
        self._completed = 0

        # Support FEXT_API_WORKERS env var override (\u00a79.2.3)
        env_workers = os.environ.get("FEXT_API_WORKERS")
        if env_workers:
            config.cache.api_workers = int(env_workers)
        api_workers = config.cache.api_workers
        qwen = QwenClient()

        # Module 2 entity extractor (multi-threaded per \u00a79.2.3)
        self.entity_extractor = Module2EntityExtractor(
            api_workers=api_workers,
            qwen_client=qwen,
        )

        # Module 3 style rewriter (multi-threaded per \u00a79.2.4)
        self.style_rewriter = Module3StyleRewriter(qwen_client=qwen, offline=offline_m3)
        self.module3_text_mode = module3_text_mode

        # GPU models (loaded lazily, protected by _gpu_lock)
        self.imagebind_model = None
        self.bert_model = None
        self.bert_tokenizer = None

        # Output files (set per-build)
        self.out_f = None
        self.retrieval_log_file = None

        # Stats tracking (protected by _stats_lock)
        self.stats = {
            "module2_empty": 0,
            "module3_empty": 0,
            "api_errors": 0,
        }

    def _load_imagebind(self):
        """Load ImageBind model (ARCHITECTURE.md \u00a73.2.2)."""
        if self.imagebind_model is None:
            ckpt_path = "models/imagebind/imagebind_huge.pth"
            model = imagebind_huge(pretrained=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt, strict=False)
            model.to(self.device)
            model.eval()
            self.imagebind_model = model

    def _load_bert(self):
        """Load BERT model for [CLS] extraction."""
        if self.bert_model is None:
            from transformers import BertModel, BertTokenizer
            model_path = "models/bert-base-multilingual-cased"
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
            self.bert_model = BertModel.from_pretrained(model_path, local_files_only=True)
            self.bert_model.to(self.device)
            self.bert_model.eval()

    def _bert_cls(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Get BERT [CLS] embeddings for a list of texts.

        Processes in batches of batch_size (\u00a74.2.5: batch=8).
        GPU-locked: only one sample accesses BERT at a time.
        """
        with self._gpu_lock:
            self._load_bert()
            all_cls = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoding = self.bert_tokenizer(
                    batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                with torch.no_grad():
                    outputs = self.bert_model(**encoding)
                all_cls.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
            return np.concatenate(all_cls, axis=0)

    # ---- Module 1: ImageBind (single-threaded GPU, \u00a79.2.6) ----

    def _encode_imagebind(self, sample: FakeNewsSample) -> Dict:
        """ImageBind encoding for a single sample (\u00a73.2.2).

        GPU-locked: only one sample accesses ImageBind at a time.
        """
        with self._gpu_lock:
            self._load_imagebind()

            # Text
            text_tokens = ib_data.load_and_transform_text([sample.text], self.device)
            text_emb = None
            if text_tokens is not None:
                with torch.no_grad():
                    out = self.imagebind_model({ModalityType.TEXT: text_tokens})
                    text_emb = out[ModalityType.TEXT].cpu().numpy()[0]

            # Video
            video_emb = None
            video_path = None
            if sample.lang == "zh":
                video_path = str(Path(self.config.data.fake_sv_root) / "filtered_video" / f"{sample.id}.mp4")
            else:
                video_path = str(Path(self.config.data.fake_tt_root) / "video" / f"{sample.id}.mp4")

            if video_path and Path(video_path).exists():
                import cv2
                frame_paths = self._extract_video_frames(video_path, self.config.cache.num_video_frames)
                if frame_paths:
                    try:
                        with torch.no_grad():
                            vision_input = ib_data.load_and_transform_vision_data(frame_paths, self.device)
                            if vision_input is not None:
                                outputs = self.imagebind_model({ModalityType.VISION: vision_input})
                                frame_embs = outputs[ModalityType.VISION]
                                video_emb = frame_embs.mean(dim=0).cpu().numpy()
                    finally:
                        for fp in frame_paths:
                            try:
                                Path(fp).unlink(missing_ok=True)
                            except OSError:
                                pass

            # Audio
            audio_emb = None
            audio_path = None
            if sample.lang == "zh":
                audio_path = str(Path(self.config.data.fake_sv_root) / "audio" / f"{sample.id}.wav")
            else:
                audio_path = str(Path(self.config.data.fake_tt_root) / "keyframe_audio" / sample.id)
                if Path(audio_path).exists():
                    wavs = sorted(Path(audio_path).glob("**/*.wav"))
                    audio_path = str(wavs[0]) if wavs else None
                else:
                    audio_path = None

            if audio_path:
                fbank = self._load_audio_tensor(audio_path)
                if fbank is not None:
                    with torch.no_grad():
                        audio_input = fbank.unsqueeze(0)
                        outputs = self.imagebind_model({ModalityType.AUDIO: audio_input})
                        audio_emb = outputs[ModalityType.AUDIO].cpu().numpy()[0]

            zero = np.zeros(1024, dtype=np.float32)
            return {
                "text_emb": text_emb if text_emb is not None else zero,
                "video_emb": video_emb if video_emb is not None else zero,
                "audio_emb": audio_emb if audio_emb is not None else zero,
            }

    def _extract_video_frames(self, video_path: str, num_frames: int = 12) -> List[str]:
        """Extract uniformly sampled frames from video."""
        import cv2
        import tempfile

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frame_paths = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=self.frame_cache_dir)
                cv2.imwrite(tmp.name, frame)
                frame_paths.append(tmp.name)
        cap.release()
        return frame_paths

    def _load_audio_tensor(self, audio_path: str):
        """Load audio as mel spectrogram for ImageBind."""
        import torchaudio

        if not audio_path or not Path(audio_path).exists():
            return None
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            target_len = 16000 * 2
            if waveform.size(1) < target_len:
                pad = target_len - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            else:
                waveform = waveform[:, :target_len]

            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=16000,
                use_energy=False, window_type="hanning",
                num_mel_bins=128, dither=0.0, frame_length=25, frame_shift=10,
            )
            fbank = fbank.transpose(0, 1)
            target_length = 204
            n_frames = fbank.size(1)
            if n_frames < target_length:
                fbank = torch.nn.functional.pad(fbank, (0, target_length - n_frames))
            else:
                fbank = fbank[:, :target_length]
            fbank = fbank.unsqueeze(0)
            mean, std = -4.268, 9.138
            fbank = (fbank - mean) / std
            return fbank.to(self.device)
        except Exception as e:
            return None

    # ---- Module 2: Multi-threaded entity extraction (\u00a79.2.3) ----

    def _compute_module2(self, sample: FakeNewsSample) -> Dict:
        """Entity extraction \u2192 BERT CLS for each of 7 types.

        Multi-threaded: ThreadPoolExecutor(7) for parallel
        Baidu search \u2192 HTTP fetch \u2192 Qwen extraction.
        """
        # Parallel entity extraction (7 threads)
        entity_results = self.entity_extractor.extract_all(sample.text, sample.lang)
        branch_texts, branch_mask = self.entity_extractor.build_branch_texts(entity_results, sample.lang)

        empty_count = sum(1 for m in branch_mask if m == 0)
        if empty_count == 7:
            with self._stats_lock:
                self.stats["module2_empty"] += 1

        # BERT encode non-empty branches (GPU, single-threaded)
        branches = np.zeros((7, 768), dtype=np.float32)
        valid_indices = [i for i, m in enumerate(branch_mask) if m == 1]
        if valid_indices:
            valid_texts = [branch_texts[i] for i in valid_indices]
            cls_vecs = self._bert_cls(valid_texts)
            for idx, vec in zip(valid_indices, cls_vecs):
                branches[idx] = vec

        # Compute 35-dim entity statistics for ablation baseline (\u00a74.2.7)
        entity_stats = self.entity_extractor.compute_entity_stats(entity_results)

        return {
            "module2_branches": branches,
            "module2_mask": np.array(branch_mask, dtype=np.float32),
            "module2_entity_stats": entity_stats,
            "entity_results": entity_results,  # full EntityResult objects for logging
        }

    # ---- Module 3: Multi-threaded style rewriting (\u00a79.2.4) ----

    def _compute_module3(self, sample: FakeNewsSample) -> Dict:
        """Style rewriting \u2192 analysis \u2192 BERT CLS.

        Multi-threaded: ThreadPoolExecutor(2) for Ta/Tb rewriting.
        Style analysis: serial after rewrite completes.
        """
        result = self.style_rewriter.rewrite_and_analyze(
            sample.text, sample.lang, text_mode=self.module3_text_mode
        )

        if not result["success"]:
            with self._stats_lock:
                self.stats["module3_empty"] += 1

        style_text = result.get("style_analysis", "")
        if not style_text:
            style_text = sample.text  # fallback if analysis failed

        # BERT CLS (GPU, single-threaded)
        bert_cls = self._bert_cls([style_text])[0]

        return {
            "module3_bert_cls": bert_cls,
            "ta_skeleton": result.get("ta_skeleton", ""),
            "tb_skeleton": result.get("tb_skeleton", ""),
        }

    # ---- Retrieval logging (\u00a74.2.8) ----

    def _serialize_entity_result(self, r) -> Dict:
        """Serialize EntityResult to JSON-compatible dict."""
        return {
            "entity_type": getattr(r, "entity_type", ""),
            "extracted_text": getattr(r, "extracted_text", ""),
            "original_entities": getattr(r, "original_entities", []),
            "is_type_missing": getattr(r, "is_type_missing", False),
            "search_urls": getattr(r, "search_urls", []),
            "source_doc_index": getattr(r, "source_doc_index", 0),
            "evidence_span": getattr(r, "evidence_span", ""),
            "success": getattr(r, "success", False),
        }

    def _write_retrieval_log(self, sample_id: str, entity_results: Dict):
        """Write EntityResult details to retrieval log file."""
        if self.retrieval_log_file is None:
            return
        record = {"id": sample_id, "results": {}}
        for etype, r in entity_results.items():
            record["results"][etype] = self._serialize_entity_result(r)
        self.retrieval_log_file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        self.retrieval_log_file.flush()

    # ---- Main processing ----

    def process_sample(self, sample: FakeNewsSample) -> Optional[Dict]:
        """Process a single sample through the full pipeline.

        Order per ARCHITECTURE.md:
          1. Text aux (CPU, fast)
          2. Module 1: ImageBind (GPU)
          3. Module 2: Entity extraction + BERT (API multi-threaded + GPU)
          4. Module 3: Style rewriting + analysis + BERT (API multi-threaded + GPU)
        """
        t0 = time.time()
        try:
            # Text auxiliary features
            text_aux = build_text_aux(sample.text, hashing_dim=self.config.cache.hashing_dim)

            # Module 1: ImageBind (GPU, \u00a79.2.6)
            ib = self._encode_imagebind(sample)
            module1_valid = 1 if (ib["text_emb"].any() or ib["video_emb"].any() or ib["audio_emb"].any()) else 0

            # Module 2: Multi-threaded entity extraction (API, \u00a79.2.3)
            m2 = self._compute_module2(sample)

            # Write retrieval log (\u00a74.2.8, thread-safe)
            with self._write_lock:
                self._write_retrieval_log(sample.id, m2["entity_results"])

            # Module 3: Multi-threaded style rewriting (API, \u00a79.2.4)
            m3 = self._compute_module3(sample)

            elapsed = time.time() - t0
            return {
                "id": sample.id,
                "label": sample.label,
                "lang": sample.lang,
                "_elapsed": round(elapsed, 1),
                "id": sample.id,
                "label": sample.label,
                "lang": sample.lang,
                # Module 1: raw ImageBind embeddings
                "module1_text_emb": ib["text_emb"].tolist(),
                "module1_video_emb": ib["video_emb"].tolist(),
                "module1_audio_emb": ib["audio_emb"].tolist(),
                "module1_valid": module1_valid,
                # Module 2: 7\u00d7768 BERT CLS vectors
                "module2_branches": m2["module2_branches"].tolist(),
                "module2_mask": m2["module2_mask"].tolist(),
                # Module 2: 35-dim entity statistics for ablation (\u00a74.2.7)
                "module2_entity_stats": m2["module2_entity_stats"].tolist(),
                # Text-only summary of entity results (full details in retrieval log)
                "entity_results": {
                    etype: r.extracted_text
                    for etype, r in m2["entity_results"].items()
                },
                # Module 3: BERT CLS from style analysis
                "module3_bert_cls": m3["module3_bert_cls"].tolist(),
                # M3 ablation: store skeletons for cache-only ablation
                "ta_skeleton": m3.get("ta_skeleton", ""),
                "tb_skeleton": m3.get("tb_skeleton", ""),
                # Text auxiliary features
                "text_aux": text_aux.tolist(),
            }

        except Exception as e:
            import traceback
            print(f"  Error processing {sample.id}: {e}")
            traceback.print_exc()
            with self._stats_lock:
                self.stats["api_errors"] += 1
            return None

    def build(self, split: str = "train", max_items: Optional[int] = None):
        """Build cache for all samples in a split."""
        self._start_time = time.time()
        self._completed = 0
        print(f"Building full feature cache for {split} split...")

        # Load dataset
        max_kwargs = {} if max_items is None else {"max_items": max_items}
        sv_dataset = FakeSVDataset(split=split, **max_kwargs)
        tt_dataset = FakeTTDataset(split=split, **max_kwargs)
        all_samples = list(sv_dataset) + list(tt_dataset)
        print(f"Total samples: {len(all_samples)}")

        # Cache file
        cache_file = self.cache_dir / f"{split}_features.jsonl"
        temp_file = self.cache_dir / f"{split}_features.jsonl.tmp"

        # Load processed IDs (resumable)
        processed_ids = set()
        if cache_file.exists():
            with open(cache_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            processed_ids.add(item["id"])
                        except json.JSONDecodeError:
                            continue
            print(f"  Already processed: {len(processed_ids)} samples")

        # Filter unprocessed
        pending = [s for s in all_samples if s.id not in processed_ids]
        print(f"  Pending: {len(pending)} samples")

        if not pending:
            print("  All samples already cached.")
            return

        # Clean up leftover temp file
        if temp_file.exists():
            temp_file.unlink()

        # Process samples with sample-level parallelism (\u00a79.2.6)
        out_f = open(temp_file, "a")
        retrieval_log_dir = self.cache_dir / "retrieval_logs"
        retrieval_log_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_log_file = open(
            retrieval_log_dir / f"{split}_retrieval_log.jsonl", "a"
        )
        total = len(pending)
        try:
            with ThreadPoolExecutor(max_workers=self.sample_workers) as executor:
                futures = [executor.submit(self.process_sample, s) for s in pending]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  Unexpected worker error: {e}")
                        with self._stats_lock:
                            self.stats["api_errors"] += 1
                        continue
                    if result is not None:
                        with self._write_lock:
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_f.flush()
                        sample_id = result.get("id", "?")
                        sample_sec = result.get("_elapsed", 0)
                    else:
                        sample_id = "FAILED"
                        sample_sec = 0
                    with self._stats_lock:
                        self._completed += 1
                        completed = self._completed
                    elapsed = time.time() - self._start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"  [{completed}/{total}] {sample_id} | "
                          f"{sample_sec:.0f}s | "
                          f"{rate:.2f}/s | ETA {eta/60:.0f}min | "
                          f"errors={self.stats['api_errors']}", flush=True)
        finally:
            out_f.close()
            if self.retrieval_log_file is not None:
                self.retrieval_log_file.close()
                self.retrieval_log_file = None

        # Merge temp file into cache
        if temp_file.exists():
            with open(cache_file, "a") as cf, open(temp_file, "r") as tf:
                for line in tf:
                    cf.write(line)
            temp_file.unlink()

        # Count final size
        line_count = 0
        with open(cache_file, "r") as f:
            for _ in f:
                line_count += 1
        print(f"  Cache saved: {cache_file} ({line_count} samples)")

        # Freeze cache to read-only if requested (\u00a77.3.5)
        if self.freeze and cache_file.exists():
            cache_file.chmod(0o444)
            print(f"  Cache frozen (read-only): {cache_file}")

    def build_all(self, split: str = "all"):
        """Build cache for train and/or test splits."""
        is_train = split in ("train", "all")
        is_test = split in ("test", "all")

        if is_train:
            max_items = self.config.data.max_train_items
            self.build("train", max_items=max_items)
        if is_test:
            max_items = self.config.data.max_test_items
            self.build("test", max_items=max_items)

        # Print stats
        print(f"\nStats: {self.stats}")

        # Verify only built cache files
        for split in ("train", "test"):
            cache_file = Path(self.config.cache.cache_dir) / f"{split}_features.jsonl"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    first = json.loads(f.readline())
                    last = None
                    count = 0
                    for line in f:
                        last = json.loads(line)
                        count += 1
                    count += 1
                print(f"\nCache verification ({split}): {count} samples, "
                      f"keys: {list(first.keys())}")
                if last:
                    print(f"  First: {first['id']}, Last: {last['id']}")

        print("\nFull cache build complete!")


def main(args):
    config = load_config(args.config)

    builder = FullFeatureCacheBuilder(config, device=args.device, workers=args.workers, freeze=args.freeze,
                                       cache_dir=args.cache_dir, module3_text_mode=args.module3_text_mode,
                                       offline_m3=args.offline_m3)
    builder.build_all(split=args.split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build full feature cache (Stage 1)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=8,
                        help="Sample-level parallelism (default: 8). GPU ops serialized via Lock.")
    parser.add_argument("--freeze", action="store_true",
                        help="Freeze cache files to read-only after build (\u00a77.3.5)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cache directory (default: from config)")
    parser.add_argument("--module3-text-mode", type=str, default="both",
                        choices=["both", "only_ta", "only_tb"],
                        help="M3 ablation: which rewritten texts to include (default: both)")
    parser.add_argument("--offline-m3", action="store_true",
                        help="Skip Qwen API calls for M3, use offline rewriting only")
    args = parser.parse_args()
    main(args)
