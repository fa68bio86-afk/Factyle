#!/usr/bin/env python3
"""T1: Module-level validation checks.

Verifies each component independently before running the full pipeline.
Logs results to both stdout and docs/validation_log.md.

Usage:
    python scripts/validate_modules.py [--gpu]

Reference: ARCHITECTURE.md \u00a71-\u00a77
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "ImageBind"))

LOG_FILE = Path(__file__).resolve().parent.parent / "docs" / "validation_log.md"


def log(msg: str, pass_fail: str = ""):
    """Print to stdout. Marker is used externally for the log."""
    ts = datetime.now().strftime("%H:%M:%S")
    marker = {"pass": "  \u2705", "fail": "  \u274c", "warn": "  \u26a0\ufe0f ", "info": "     "}.get(pass_fail, "     ")
    print(f"{marker} {msg}")


def append_log(tier: str, section: str, result: str, detail: str = ""):
    """Append a row to the validation log."""
    line = f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} | {tier} | {section} | {result} | {detail} |\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)


# =============================================================================
# Checks
# =============================================================================


def check_environment():
    log("--- Environment ---", "info")
    issues = []

    # Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    log(f"Python {py_ver}")
    if sys.version_info < (3, 8):
        issues.append("Python < 3.8")

    # PyTorch
    try:
        import torch
        log(f"PyTorch {torch.__version__}")
        cuda = torch.cuda.is_available()
        if cuda:
            log(f"CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}")
        else:
            log("CUDA not available (CPU mode)", "warn")
    except ImportError:
        log("torch not installed", "fail")
        issues.append("torch import failed")

    # spaCy
    try:
        import spacy
        log(f"spaCy {spacy.__version__}")
        for model_name in ["zh_core_web_sm", "en_core_web_sm"]:
            try:
                spacy.load(model_name)
                log(f"  {model_name} loaded")
            except OSError:
                log(f"  {model_name} not found (will use regex fallback)", "warn")
    except ImportError:
        log("spaCy not installed", "warn")

    # Key dependencies
    for mod_name, label in [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("torchaudio", "torchaudio"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
        ("yaml", "pyyaml"),
    ]:
        try:
            __import__(mod_name)
            log(f"{label} OK")
        except ImportError:
            log(f"{label} not found", "fail" if mod_name in ("numpy", "torchaudio", "transformers") else "warn")
            if mod_name in ("numpy", "torchaudio", "transformers"):
                issues.append(f"{mod_name} missing")

    if issues:
        log(f"Environment issues: {', '.join(issues)}", "fail")
        return False
    log("Environment OK", "pass")
    return True


def check_config():
    log("--- Config Loading ---", "info")
    try:
        from factyle.utils.config import load_config
        cfg = load_config(str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"))
        log(f"  output_dir: {cfg.output_dir}")
        log(f"  hashing_dim: {cfg.cache.hashing_dim}")
        log(f"  module_output_dim: {cfg.model.module_output_dim}")
        log(f"  api_workers: {cfg.cache.api_workers}")
        assert cfg.cache.hashing_dim == 512, f"hashing_dim should be 512, got {cfg.cache.hashing_dim}"
        log("Config OK", "pass")
        return True
    except Exception as e:
        log(f"Config failed: {e}", "fail")
        return False


def check_datasets():
    log("--- Dataset Loading ---", "info")
    try:
        from factyle.data.dataset import FakeSVDataset, FakeTTDataset

        sv = FakeSVDataset(split="train", max_items=2)
        tt = FakeTTDataset(split="train", max_items=2)
        log(f"FakeSV train: {len(sv)} samples (requested 2)")
        log(f"FakeTT train: {len(tt)} samples (requested 2)")

        s = sv[0]
        log(f"  Sample keys: id={s.id}, label={s.label}, lang={s.lang}")
        log(f"  Text length: {len(s.text)} chars")
        assert hasattr(s, "text"), "Sample missing 'text' property"
        assert s.lang in ("zh", "en"), f"Unexpected lang: {s.lang}"

        sv_test = FakeSVDataset(split="test", max_items=2)
        tt_test = FakeTTDataset(split="test", max_items=2)
        log(f"FakeSV+TT test total: {len(sv_test) + len(tt_test)} samples")

        log("Datasets OK", "pass")
        return True
    except Exception as e:
        log(f"Datasets failed: {e}", "fail")
        return False


def check_text_aux():
    log("--- Text Auxiliary Features (\u00a76.3) ---", "info")
    try:
        from factyle.features.text import build_text_aux

        feat = build_text_aux("This is a test news item with numbers 123.", hashing_dim=512)
        log(f"  text_aux shape: {feat.shape}")
        assert feat.shape == (524,), f"Expected (524,), got {feat.shape}"
        assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"
        log(f"  hashing norm: {np.linalg.norm(feat[:512]):.4f}")
        log("text_aux OK", "pass")
        return True
    except Exception as e:
        log(f"text_aux failed: {e}", "fail")
        return False


def check_model():
    log("--- Model Initialization & Forward (\u00a72-\u00a76) ---", "info")
    try:
        import torch
        from factyle.models.torch_models import FactStyleFusionClassifier

        device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        model = FactStyleFusionClassifier(
            module_output_dim=256,
            mlp2_hidden=1024,
            text_aux_dim=64,
            text_aux_input_dim=524,
            dropout=0.1,
        ).to(device)
        model.eval()

        B = 4
        dummy = {
            "text_emb": torch.randn(B, 1024).to(device),
            "video_emb": torch.randn(B, 1024).to(device),
            "audio_emb": torch.randn(B, 1024).to(device),
            "branch_bert_cls": torch.randn(B, 7, 768).to(device),
            "branch_mask": torch.ones(B, 7).to(device),
            "style_bert_cls": torch.randn(B, 768).to(device),
            "text_aux": torch.randn(B, 524).to(device),
        }

        with torch.no_grad():
            # Forward each module individually
            fint = model.forward_module1(dummy["text_emb"], dummy["video_emb"], dummy["audio_emb"])
            fext = model.forward_module2(dummy["branch_bert_cls"], dummy["branch_mask"])
            fstyle = model.forward_module3(dummy["style_bert_cls"])
            logits = model.forward_fusion(fint, fext, fstyle, dummy["text_aux"])

        log(f"  FINT_FACT: {fint.shape}  (expected (B, 256))")
        log(f"  FEXT_FACT: {fext.shape}  (expected (B, 256))")
        log(f"  FSTYLE:    {fstyle.shape}  (expected (B, 256))")
        log(f"  Logits:    {logits.shape}  (expected (B, 1))")

        assert fint.shape == (B, 256), f"FINT wrong: {fint.shape}"
        assert fext.shape == (B, 256), f"FEXT wrong: {fext.shape}"
        assert fstyle.shape == (B, 256), f"FSTYLE wrong: {fstyle.shape}"
        assert logits.shape == (B, 1), f"Logits wrong: {logits.shape}"

        # Test missing modality (all zeros \u2192 FINT should be zero)
        zero_emb = torch.zeros(B, 1024).to(device)
        fint_zero = model.forward_module1(zero_emb, zero_emb, zero_emb)
        assert fint_zero.abs().sum() == 0, "Missing modality should give zero FINT_FACT"

        # Test absent embeddings (mask=0 \u2192 should use absent_embedding)
        zero_mask = torch.zeros(B, 7).to(device)
        fext_absent = model.forward_module2(dummy["branch_bert_cls"], zero_mask)
        assert fext_absent.shape == (B, 256), f"Absent FEXT wrong: {fext_absent.shape}"

        # Test end-to-end forward
        logits_e2e = model(
            dummy["text_emb"], dummy["video_emb"], dummy["audio_emb"],
            dummy["branch_bert_cls"], dummy["branch_mask"],
            dummy["style_bert_cls"], dummy["text_aux"],
        )
        assert logits_e2e.shape == (B, 1), f"E2E logits wrong: {logits_e2e.shape}"
        assert torch.allclose(logits_e2e, logits), "E2E forward should match modular forward"

        log("Model OK", "pass")
        return True
    except Exception as e:
        import traceback
        log(f"Model failed: {e}", "fail")
        traceback.print_exc()
        return False


def check_imagebind():
    log("--- ImageBind Checkpoint (\u00a73.2.2) ---", "info")
    ckpt_path = Path(__file__).resolve().parent.parent / "models" / "imagebind" / "imagebind_huge.pth"
    if not ckpt_path.exists():
        log(f"Checkpoint not found: {ckpt_path}", "fail")
        return False

    log(f"  Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1024**3:.1f} GB)")

    # Try symlink resolution
    real = ckpt_path.resolve()
    log(f"  Resolves to: {real}")
    if not real.exists():
        log("  Symlink broken!", "fail")
        return False

    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        log(f"  Loaded: {len(ckpt)} keys")
        log("ImageBind checkpoint OK", "pass")
        return True
    except Exception as e:
        log(f"  Checkpoint load failed: {e}", "fail")
        return False


def check_bert():
    log("--- BERT Model (\u00a74.2.5) ---", "info")
    model_path = Path(__file__).resolve().parent.parent / "models" / "bert-base-multilingual-cased"
    if not model_path.exists():
        log(f"BERT not found: {model_path}", "fail")
        return False

    log(f"  Path: {model_path}")
    real = model_path.resolve()
    log(f"  Resolves to: {real}")

    try:
        from transformers import BertModel, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = BertModel.from_pretrained(str(model_path), local_files_only=True)
        import torch
        model.eval()
        inputs = tokenizer(["test sentence"], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
        cls_vec = out.last_hidden_state[:, 0, :]
        log(f"  BERT CLS shape: {cls_vec.shape}  (expected (1, 768))")
        assert cls_vec.shape == (1, 768), f"BERT CLS wrong: {cls_vec.shape}"
        log("BERT OK", "pass")
        return True
    except Exception as e:
        log(f"BERT failed: {e}", "fail")
        return False


def check_env_file():
    log("--- API Keys (.env) ---", "info")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        log(".env file not found", "fail")
        return False
    log(f"  .env exists ({env_path.stat().st_size} bytes)")

    keys_found = 0
    for key_name in ["BAIDU_SEARCH_API_KEY", "QWEN_API_KEY"]:
        val = None
        with open(env_path) as f:
            for line in f:
                if line.startswith(key_name):
                    val = line.strip().split("=", 1)[1].strip()
                    break
        if val and "your_" not in val:
            log(f"  {key_name}: set")
            keys_found += 1
        else:
            log(f"  {key_name}: missing or placeholder", "fail")
    if keys_found == 2:
        log("API keys OK", "pass")
        return True
    return False


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T1: Module-level validation checks")
    parser.add_argument("--gpu", action="store_true", help="Run GPU-dependent checks")
    args = parser.parse_args()

    # Import numpy here (after path setup)
    import numpy as np

    print("\n" + "=" * 60)
    print("  T1: Module-Level Validation Checks")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.gpu:
        print("  GPU checks: ENABLED")
    print("=" * 60)
    print()

    checks = [
        ("Environment", check_environment),
        ("Config Loading", check_config),
        ("Datasets", check_datasets),
        ("text_aux (\u00a76.3)", check_text_aux),
        ("Model (\u00a72-\u00a76)", check_model),
        ("ImageBind checkpoint", check_imagebind),
        ("BERT model", check_bert),
        ("API keys (.env)", check_env_file),
    ]

    results = {}
    all_pass = True
    for name, fn in checks:
        try:
            ok = fn()
        except Exception as e:
            print(f"  UNEXPECTED ERROR: {e}")
            ok = False
        results[name] = "\u2705 PASS" if ok else "\u274c FAIL"
        if not ok:
            all_pass = False
        print()

    # Summary
    print("=" * 60)
    print("  T1 Summary")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {status}  {name}")
    print(f"\n  Overall: {'\u2705 ALL PASS' if all_pass else '\u274c SOME FAILED'}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # Write to validation log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    overall = "\u2705 PASS" if all_pass else "\u274c FAIL"
    detail = "; ".join(f"{n}: {s.split()[0]}" for n, s in results.items())
    append_log("T1", "All module checks", overall, detail)

    sys.exit(0 if all_pass else 1)
