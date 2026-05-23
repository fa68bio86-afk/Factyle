#!/usr/bin/env python3
"""Stage 2: Train from cached intermediate representations.

Reference: ARCHITECTURE.md \u00a77.1, \u00a77.4.2

Supports:
  - Hold-out validation split (\u00a77.4.2)
  - Z-score feature normalization (\u00a77.1)
  - Validation-based early stopping
  - Per-dataset metrics (FakeSV zh / FakeTT en) \u2014 \u00a78
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from factyle.models.torch_models import FactStyleFusionClassifier
from factyle.utils.config import load_config


class CachedFeatureDataset(Dataset):
    """Load cached intermediate representations for training.

    Each sample contains (\u00a77.1):
      - module1_text/video/audio_emb: (1024,) ImageBind embeddings
      - module1_valid: 0/1
      - module2_branches: (7, 768) BERT CLS vectors
      - module2_mask: (7,) presence mask
      - module3_bert_cls: (768,) BERT CLS from style analysis
      - text_aux: (524,) hashing + text_stats
      - label: 0/1
      - lang: "zh"/"en"
    """

    def __init__(
        self,
        cache_path: str,
        lang_filter: str = "all",
        teacher_probs_path: str = None,
    ):
        teacher_by_id = {}
        teacher_by_pos = []
        if teacher_probs_path:
            with open(teacher_probs_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    prob = float(item.get("teacher_prob", item.get("prob")))
                    teacher_by_pos.append(prob)
                    if "id" in item:
                        teacher_by_id[str(item["id"])] = prob

        self.samples = []
        with open(cache_path, "r") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if line:
                    s = json.loads(line)
                    if lang_filter == "all" or s.get("lang", "zh") == lang_filter:
                        teacher_prob = teacher_by_id.get(str(s.get("id")))
                        if teacher_prob is None and line_idx < len(teacher_by_pos):
                            teacher_prob = teacher_by_pos[line_idx]
                        if teacher_prob is not None:
                            s["_teacher_prob"] = teacher_prob
                        self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "id": s["id"],
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "lang": s.get("lang", "zh"),
            # Module 1: raw ImageBind embeddings (ARCHITECTURE.md \u00a73.2.2)
            "module1_text_emb": torch.tensor(s["module1_text_emb"], dtype=torch.float32),
            "module1_video_emb": torch.tensor(s["module1_video_emb"], dtype=torch.float32),
            "module1_audio_emb": torch.tensor(s["module1_audio_emb"], dtype=torch.float32),
            "module1_valid": torch.tensor(s.get("module1_valid", 1), dtype=torch.float32),
            # Module 2: 7\u00d7768 BERT CLS
            "module2_branches": torch.tensor(s["module2_branches"], dtype=torch.float32),
            "module2_mask": torch.tensor(s["module2_mask"], dtype=torch.float32),
            # Module 3: 768 BERT CLS
            "module3_bert_cls": torch.tensor(s["module3_bert_cls"], dtype=torch.float32),
            # Hashing text
            "text_aux": torch.tensor(s["text_aux"], dtype=torch.float32),
            # Module 2 entity statistics (5 stats \u00d7 7 types = 35 dims)
            "entity_stats": torch.tensor(s.get("module2_entity_stats", [0]*35), dtype=torch.float32),
        }
        if "_teacher_prob" in s:
            item["teacher_prob"] = torch.tensor(s["_teacher_prob"], dtype=torch.float32)
        return item


class FeatureNormalizer:
    """Z-score normalization for cached features (\u00a77.1).

    Only normalizes float feature vectors, NOT binary/mask fields
    (module1_valid, module2_mask).
    """

    def __init__(self):
        self.stats = {}  # key -> (mean, std)

    def fit(self, dataset: CachedFeatureDataset):
        """Compute per-feature mean/std from all samples."""
        feature_keys = [
            "module1_text_emb", "module1_video_emb", "module1_audio_emb",
            "module2_branches", "module3_bert_cls", "text_aux",
            # entity_stats excluded: already normalized statistics (counts, ratios)
        ]

        # Collect all values
        all_vals = {k: [] for k in feature_keys}
        for i in range(len(dataset)):
            item = dataset[i]
            for k in feature_keys:
                all_vals[k].append(item[k].numpy())

        # Compute stats
        for k, vals in all_vals.items():
            arr = np.stack(vals, axis=0)  # (N, D)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)  # avoid div-by-zero
            self.stats[k] = (mean, std)

        print(f"  Normalizer: computed stats for {len(feature_keys)} feature groups")

    def transform(self, item: dict) -> dict:
        """Apply z-score normalization to a single item."""
        result = dict(item)
        for k, (mean, std) in self.stats.items():
            if k in result:
                arr = result[k].numpy()
                result[k] = torch.tensor((arr - mean) / std, dtype=torch.float32)
        return result


class NormalizedDataset(Dataset):
    """Wrapper applying z-score normalization on-the-fly (\u00a77.1)."""

    def __init__(self, base: Dataset, normalizer: FeatureNormalizer):
        self.base = base
        self.normalizer = normalizer

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.normalizer.transform(self.base[idx])


def create_validation_split(
    dataset: CachedFeatureDataset, val_ratio: float = 0.2, seed: int = 42
) -> tuple:
    """Create stratified train/val split from dataset (\u00a77.4.2).

    Uses StratifiedShuffleSplit to maintain label distribution.

    Returns:
        (train_indices, val_indices)
    """
    labels = [dataset[i]["label"].item() for i in range(len(dataset))]
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed
    )
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    return train_idx, val_idx


def compute_metrics(labels, preds, prefix=""):
    """Compute all 8 metrics. Reference: ARCHITECTURE.md \u00a78."""
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    prec_fake, recall_fake, f1_fake, _ = precision_recall_fscore_support(
        labels, preds, labels=[1], average=None, zero_division=0
    )
    prec_real, recall_real, f1_real, _ = precision_recall_fscore_support(
        labels, preds, labels=[0], average=None, zero_division=0
    )

    return {
        f"{prefix}accuracy": float(acc),
        f"{prefix}macro_f1": float(macro_f1),
        f"{prefix}fake_precision": float(prec_fake[0]) if len(prec_fake) else 0.0,
        f"{prefix}fake_recall": float(recall_fake[0]) if len(recall_fake) else 0.0,
        f"{prefix}fake_f1": float(f1_fake[0]) if len(f1_fake) else 0.0,
        f"{prefix}real_precision": float(prec_real[0]) if len(prec_real) else 0.0,
        f"{prefix}real_recall": float(recall_real[0]) if len(recall_real) else 0.0,
        f"{prefix}real_f1": float(f1_real[0]) if len(f1_real) else 0.0,
    }


def add_per_dataset_metrics(metrics, labels, preds, langs, prefix=""):
    """Add per-dataset (FakeSV zh / FakeTT en) metrics to dict.

    Reference: ARCHITECTURE.md \u00a78
    """
    sv_l = [l for l, lang in zip(labels, langs) if lang == "zh"]
    sv_p = [p for p, lang in zip(preds, langs) if lang == "zh"]
    tt_l = [l for l, lang in zip(labels, langs) if lang == "en"]
    tt_p = [p for p, lang in zip(preds, langs) if lang == "en"]
    if sv_l:
        metrics.update(compute_metrics(sv_l, sv_p, prefix=f"{prefix}sv_"))
    if tt_l:
        metrics.update(compute_metrics(tt_l, tt_p, prefix=f"{prefix}tt_"))
    return metrics


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    FL(p_t) = -\u03b1_t * (1-p_t)^\u03b3 * log(p_t)
    Focuses training on hard examples by down-weighting easy ones.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha >= 0:
            alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_weight
        losses = focal_weight * bce
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()


class LabelSmoothingBCE(nn.Module):
    """BCE with label smoothing.

    Replaces hard 0/1 targets with alpha/1-alpha, reducing overconfidence.
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, target, reduction=self.reduction)


def collate_fn(batch):
    result = {}
    keys = batch[0].keys()
    for k in keys:
        if k in ("lang", "id"):
            result[k] = [b[k] for b in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
        else:
            result[k] = torch.tensor([b[k] for b in batch])
    return result


def langs_to_ids(langs, device):
    return torch.tensor([0 if lang == "zh" else 1 for lang in langs],
                        dtype=torch.long, device=device)


def lang_loss_weights(langs, weights, device):
    if weights is None:
        return None
    return torch.tensor(
        [weights.get(lang, 1.0) for lang in langs],
        dtype=torch.float32,
        device=device,
    )


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    losses = losses.view(-1)
    if weights is None:
        return losses.mean()
    weights = weights.view(-1).to(losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def load_checkpoint_state(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} did not contain a state dict")
    if checkpoint and all(str(k).startswith("module.") for k in checkpoint.keys()):
        checkpoint = {str(k)[7:]: v for k, v in checkpoint.items()}
    return checkpoint


def load_model_checkpoint(model, checkpoint_path: str, strict: bool = False):
    state = load_checkpoint_state(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=strict)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(
        f"  Init checkpoint: loaded {checkpoint_path} "
        f"(strict={strict}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        print(f"    Missing keys sample: {missing[:8]}")
    if unexpected:
        print(f"    Unexpected keys sample: {unexpected[:8]}")


def freeze_bert_after_load(model):
    bert_model = getattr(model, "bert_model", None)
    if bert_model is None:
        print("  Freeze BERT requested, but this model has no trainable BERT module")
        return
    frozen = 0
    for param in bert_model.parameters():
        frozen += param.numel()
        param.requires_grad = False
    print(f"  BERT frozen after checkpoint load ({frozen:,} params)")


def compact_checkpoint_state(model) -> dict:
    state = model.state_dict()
    if getattr(model, "bert_model", None) is None or getattr(model, "lora_rank", 0) <= 0:
        return {k: v.cpu() for k, v in state.items()}
    compact = {}
    for key, value in state.items():
        if not key.startswith("bert_model.") or ".lora_" in key:
            compact[key] = value.cpu()
    skipped = len(state) - len(compact)
    print(f"  Compact checkpoint: saving {len(compact)} tensors, skipped {skipped} frozen BERT tensors")
    return compact


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    ema_model=None,
    ema_decay=0,
    mixup_alpha=0,
    distill_alpha=0.0,
    distill_temperature=1.0,
    language_loss_weights=None,
    text_lookup=None,
    bert_tokenizer=None,
):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in loader:
        labels = batch["label"].to(device)
        text_aux = batch["text_aux"].to(device)

        m1_text = batch["module1_text_emb"].to(device)
        m1_video = batch["module1_video_emb"].to(device)
        m1_audio = batch["module1_audio_emb"].to(device)
        m2_branches = batch["module2_branches"].to(device)
        m2_mask = batch["module2_mask"].to(device)

        # When training BERT, use raw text instead of cached bert_cls
        use_train_bert = getattr(model, 'train_bert', False) and text_lookup is not None and bert_tokenizer is not None
        if use_train_bert:
            texts = [text_lookup.get(sid, "") for sid in batch["id"]]
            encoded = bert_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            bert_input_ids = encoded["input_ids"].to(device)
            bert_attention_mask = encoded["attention_mask"].to(device)
        else:
            m3_bert = batch["module3_bert_cls"].to(device)

        entity_stats = batch.get("entity_stats")
        if entity_stats is not None:
            entity_stats = entity_stats.to(device)
        lang_ids = langs_to_ids(batch["lang"], device) if (getattr(model, "language_aware", False) or getattr(model, "dann_alpha", 0) > 0) else None
        teacher_probs = batch.get("teacher_prob")
        if teacher_probs is not None:
            teacher_probs = teacher_probs.to(device)
        sample_weights = lang_loss_weights(batch["lang"], language_loss_weights, device)

        # Mixup: \u03bb ~ Beta(\u03b1, \u03b1), mix features and labels
        if mixup_alpha > 0:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample()
            perm = torch.randperm(labels.size(0), device=device)
            m1_text = lam * m1_text + (1 - lam) * m1_text[perm]
            m1_video = lam * m1_video + (1 - lam) * m1_video[perm]
            m1_audio = lam * m1_audio + (1 - lam) * m1_audio[perm]
            m2_branches = lam * m2_branches + (1 - lam) * m2_branches[perm]
            m2_mask = torch.maximum(m2_mask, m2_mask[perm])
            if not use_train_bert:
                m3_bert = lam * m3_bert + (1 - lam) * m3_bert[perm]
            text_aux = lam * text_aux + (1 - lam) * text_aux[perm]
            if entity_stats is not None:
                entity_stats = lam * entity_stats + (1 - lam) * entity_stats[perm]
            if teacher_probs is not None:
                teacher_probs = lam * teacher_probs + (1 - lam) * teacher_probs[perm]
            if sample_weights is not None:
                sample_weights = lam * sample_weights + (1 - lam) * sample_weights[perm]
            mix_labels = labels[perm]
        else:
            lam = 1.0
            mix_labels = None

        fint = model.forward_module1(m1_text, m1_video, m1_audio)
        fext = model.forward_module2(m2_branches, m2_mask, entity_stats=entity_stats)
        if use_train_bert:
            fstyle = model.forward_module3(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        else:
            fstyle = model.forward_module3(style_bert_cls=m3_bert)
        logits = model.forward_fusion(
            fint, fext, fstyle, text_aux,
            entity_stats=entity_stats, lang_ids=lang_ids,
        )

        flat_logits = logits.view(-1)
        if mixup_alpha > 0:
            hard_loss = lam * criterion(flat_logits, labels) + (1 - lam) * criterion(flat_logits, mix_labels)
        else:
            hard_loss = criterion(flat_logits, labels)

        if teacher_probs is not None and distill_alpha > 0:
            temp = max(float(distill_temperature), 1e-6)
            soft_targets = teacher_probs.clamp(0.0, 1.0)
            distill_loss = F.binary_cross_entropy_with_logits(
                flat_logits / temp, soft_targets, reduction="none"
            ) * (temp ** 2)
            loss_values = (1 - distill_alpha) * hard_loss + distill_alpha * distill_loss
        else:
            loss_values = hard_loss
        loss = weighted_mean(loss_values, sample_weights)

        # DANN: language adversarial loss (gradient reversal)
        if getattr(model, "dann_alpha", 0) > 0 and model.training and lang_ids is not None:
            dann_logits = model.get_dann_logits(lang_ids)
            if dann_logits is not None:
                loss = loss + F.cross_entropy(dann_logits, lang_ids)

        optimizer.zero_grad()
        loss.backward()
        if hasattr(model, 'grad_clip_val') and model.grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip_val)
        optimizer.step()

        # EMA update after each step (mutates shared dict)
        if ema_model is not None and ema_decay > 0:
            if not ema_model:
                for k, v in model.state_dict().items():
                    ema_model[k] = v.clone().detach()
            else:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_model[k] = ema_decay * ema_model[k] + (1 - ema_decay) * v.detach()

        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        threshold = getattr(model, 'classifier_output_threshold', 0.5)
        preds = (probs >= threshold).astype(int)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.flatten())

    return total_loss / len(loader), all_labels, all_preds


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresholds=None, return_probs=False,
             text_lookup=None, bert_tokenizer=None):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_langs = []
    all_probs = []

    for batch in loader:
        labels = batch["label"].to(device)
        text_aux = batch["text_aux"].to(device)

        m1_text = batch["module1_text_emb"].to(device)
        m1_video = batch["module1_video_emb"].to(device)
        m1_audio = batch["module1_audio_emb"].to(device)
        fint = model.forward_module1(m1_text, m1_video, m1_audio)

        m2_branches = batch["module2_branches"].to(device)
        m2_mask = batch["module2_mask"].to(device)
        entity_stats = batch.get("entity_stats")
        if entity_stats is not None:
            entity_stats = entity_stats.to(device)
        lang_ids = langs_to_ids(batch["lang"], device) if getattr(model, "language_aware", False) else None
        fext = model.forward_module2(m2_branches, m2_mask, entity_stats=entity_stats)

        use_train_bert = getattr(model, 'train_bert', False) and text_lookup is not None and bert_tokenizer is not None
        if use_train_bert:
            texts = [text_lookup.get(sid, "") for sid in batch["id"]]
            encoded = bert_tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            bert_input_ids = encoded["input_ids"].to(device)
            bert_attention_mask = encoded["attention_mask"].to(device)
            fstyle = model.forward_module3(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
        else:
            m3_bert = batch["module3_bert_cls"].to(device)
            fstyle = model.forward_module3(style_bert_cls=m3_bert)

        logits = model.forward_fusion(
            fint, fext, fstyle, text_aux,
            entity_stats=entity_stats, lang_ids=lang_ids,
        )
        loss_values = criterion(logits.view(-1), labels)
        loss = loss_values.mean() if loss_values.dim() > 0 else loss_values

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        base_threshold = getattr(model, 'classifier_output_threshold', 0.5)
        if isinstance(thresholds, dict):
            batch_thresholds = np.array(
                [thresholds.get(lang, thresholds.get("all", base_threshold))
                 for lang in batch["lang"]],
                dtype=np.float32,
            )
            preds = (probs >= batch_thresholds).astype(int)
        else:
            threshold = base_threshold if thresholds is None else thresholds
            preds = (probs >= threshold).astype(int)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.flatten())
        all_langs.extend(batch["lang"])
        all_probs.extend(probs.tolist())

    if return_probs:
        return total_loss / len(loader), all_labels, all_preds, all_langs, all_probs
    return total_loss / len(loader), all_labels, all_preds, all_langs


def _score_for_threshold(labels, probs, threshold, metric):
    preds = (np.asarray(probs) >= threshold).astype(int)
    if metric == "macro_f1":
        return f1_score(labels, preds, average="macro", zero_division=0)
    return accuracy_score(labels, preds)


def tune_thresholds(labels, probs, langs, metric="accuracy", per_lang=False):
    """Tune decision threshold(s) on validation predictions."""
    grid = np.arange(0.05, 0.951, 0.005)
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    langs = np.asarray(langs)

    def best_for_mask(mask):
        best_threshold = 0.5
        best_score = -1.0
        for threshold in grid:
            score = _score_for_threshold(labels[mask], probs[mask], threshold, metric)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold, float(best_score)

    if per_lang:
        thresholds = {}
        scores = {}
        for lang in sorted(set(langs.tolist())):
            mask = langs == lang
            thresholds[lang], scores[lang] = best_for_mask(mask)
        return thresholds, scores

    threshold, score = best_for_mask(np.ones(len(labels), dtype=bool))
    return {"all": threshold}, {"all": score}


def apply_overrides(config, args):
    """Apply CLI overrides to config object (mutates in place)."""
    if args.cache_dir is not None:
        config.cache.cache_dir = args.cache_dir
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.module_output_dim is not None:
        config.model.module_output_dim = args.module_output_dim
    if args.mlp2_hidden is not None:
        config.model.mlp2_hidden = args.mlp2_hidden
    if args.text_aux_dim is not None:
        config.model.text_aux_dim = args.text_aux_dim
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.patience is not None:
        config.training.patience = args.patience
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.training.seed = args.seed


def main(args):
    config = load_config(args.config)
    apply_overrides(config, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set seed
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Load cached features
    cache_dir = Path(config.cache.cache_dir)
    test_lang = args.test_lang if args.test_lang else args.lang_filter
    train_dataset = CachedFeatureDataset(
        str(cache_dir / "train_features.jsonl"),
        lang_filter=args.lang_filter,
        teacher_probs_path=args.teacher_probs,
    )
    test_dataset = CachedFeatureDataset(str(cache_dir / "test_features.jsonl"), lang_filter=test_lang)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    if args.teacher_probs:
        n_teacher = sum(1 for s in train_dataset.samples if "_teacher_prob" in s)
        print(f"  Teacher probs: {n_teacher}/{len(train_dataset)} train samples")

    # English oversampling: duplicate English samples to balance the 81% zh / 19% en split
    if args.en_oversample > 1:
        en_samples = [s for s in train_dataset.samples if s.get("lang") == "en"]
        en_count = len(en_samples)
        if en_count > 0:
            for _ in range(args.en_oversample - 1):
                import copy
                train_dataset.samples.extend(copy.deepcopy(en_samples))
            print(f"  English oversample {args.en_oversample}x: {en_count} \u2192 {en_count * args.en_oversample} en "
                  f"(total: {len(train_dataset)} samples)")

    # Level 5 E1: subsample training data BEFORE validation split
    if args.data_fraction < 1.0:
        n_total = len(train_dataset)
        n_keep = max(1, int(n_total * args.data_fraction))
        rng = np.random.default_rng(config.training.seed)
        keep_idx = rng.choice(n_total, n_keep, replace=False)
        train_dataset = Subset(train_dataset, keep_idx)
        print(f"  Data fraction {args.data_fraction}: {n_total} \u2192 {n_keep} samples")

    # Validation split (\u00a77.4.2)
    val_dataset = None
    if args.val_split > 0.0:
        val_split_seed = args.val_seed if args.val_seed is not None else config.training.seed
        train_idx, val_idx = create_validation_split(
            train_dataset, val_ratio=args.val_split, seed=val_split_seed
        )
        val_dataset = Subset(train_dataset, val_idx)
        train_dataset = Subset(train_dataset, train_idx)
        print(f"  Validation split: {len(train_dataset)} train, {len(val_dataset)} val")

    # Z-score normalization (\u00a77.1): fit ONLY on training samples (no val info leak)
    normalizer = FeatureNormalizer()
    normalizer.fit(train_dataset)

    # Create data loaders (all datasets wrapped with NormalizedDataset)
    train_loader = DataLoader(
        NormalizedDataset(train_dataset, normalizer),
        batch_size=config.training.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            NormalizedDataset(val_dataset, normalizer),
            batch_size=config.training.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2
        )
    test_loader = DataLoader(
        NormalizedDataset(test_dataset, normalizer),
        batch_size=config.training.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )

    # text_aux input dim = hashing_dim + 12 text_stats (\u00a76.3)
    text_aux_input_dim = config.cache.hashing_dim + 12
    entity_stats_dim = getattr(config.model, 'entity_stats_dim', args.entity_stats_dim)
    model = FactStyleFusionClassifier(
        module_output_dim=config.model.module_output_dim,
        mlp2_hidden=config.model.mlp2_hidden,
        text_aux_dim=config.model.text_aux_dim,
        text_aux_input_dim=text_aux_input_dim,
        dropout=config.model.dropout,
        entity_stats_dim=entity_stats_dim,
        ablate_module1=args.ablate_module1,
        ablate_module2=args.ablate_module2,
        ablate_module3=args.ablate_module3,
        ablate_text_aux=args.ablate_text_aux,
        simple_module1=args.simple_module1,
        simple_module2=args.simple_module2,
        simple_module3=args.simple_module3,
        simple_fusion=args.simple_fusion,
        weak_module1=args.weak_module1,
        no_co_attn=args.no_co_attn,
        weak_fusion=args.weak_fusion,
        entity_conditioned=args.entity_conditioned,
        ablate_entity_gating=args.ablate_entity_gating,
        deep_fusion=args.deep_fusion,
        num_heads=args.num_heads,
        language_aware=args.language_aware,
        lang_emb_dim=args.lang_emb_dim,
        per_lang_classifier=args.per_lang_classifier,
        dann_alpha=args.dann_alpha,
        train_bert=args.train_bert,
        bert_tune_layers=args.bert_tune_layers,
        lora_rank=args.lora_rank,
        bert_pool=args.bert_pool,
        fact_fusion_extra=args.fact_fusion_extra,
    ).to(device)
    model.classifier_output_threshold = config.training.decision_threshold
    model.grad_clip_val = args.grad_clip

    if args.init_checkpoint:
        load_model_checkpoint(
            model,
            args.init_checkpoint,
            strict=args.init_checkpoint_strict,
        )
    if args.freeze_bert_after_load:
        freeze_bert_after_load(model)

    # Loss and optimizer
    ls = args.label_smoothing
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="none")
        print(f"  Loss: Focal (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    elif ls > 0:
        criterion = LabelSmoothingBCE(smoothing=ls, reduction="none")
        print(f"  Loss: BCE + LabelSmoothing({ls})")
    else:
        pos_weight = torch.tensor([args.pos_weight], device=device) if args.pos_weight > 0 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        w_str = f", pos_weight={args.pos_weight}" if args.pos_weight > 0 else ""
        print(f"  Loss: BCE{w_str}")

    language_loss_weights = None
    if args.zh_loss_weight != 1.0 or args.en_loss_weight != 1.0:
        language_loss_weights = {"zh": args.zh_loss_weight, "en": args.en_loss_weight}
        print(f"  Language loss weights: {language_loss_weights}")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in model.parameters())
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying freeze flags")
    print(f"  Trainable params: {trainable_count:,}/{total_count:,}")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    else:  # adamw (default)
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    # Learning rate scheduler (with optional warmup)
    scheduler = None
    if args.scheduler == "cosine":
        warmup_epochs = args.warmup_ratio * config.training.epochs
        if warmup_epochs > 0:
            def warmup_cosine_lr(epoch):
                if epoch < warmup_epochs:
                    return epoch / max(1, warmup_epochs)
                progress = (epoch - warmup_epochs) / max(1, config.training.epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr)
            print(f"  Scheduler: Warmup({warmup_epochs:.1f} epochs) + Cosine")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.training.epochs, eta_min=1e-6
            )
    elif args.scheduler == "onecycle":
        wu = max(args.warmup_ratio, 0.05)  # default 5% warmup if ratio=0
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.training.learning_rate,
            total_steps=config.training.epochs, pct_start=wu,
            anneal_strategy="cos", final_div_factor=100,
        )
        print(f"  Scheduler: OneCycleLR (warmup={wu:.0%}, max_lr={config.training.learning_rate:.2e})")

    # EMA (Exponential Moving Average)
    ema_state = None
    ema_decay = args.ema_decay
    if ema_decay > 0:
        ema_state = {}
        print(f"  EMA: decay={ema_decay}")

    # Load raw text lookup for trainable BERT
    text_lookup = {}
    bert_tokenizer = None
    if args.train_bert:
        from transformers import BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained("models/bert-base-multilingual-cased")
        for split_name in ["train", "test"]:
            for dataset_root in [config.data.fake_sv_root, config.data.fake_tt_root]:
                path = Path(dataset_root) / f"{split_name}_title_transcript.json"
                if path.exists():
                    import json as _json
                    with open(path) as f:
                        data = _json.load(f)
                    for item in data:
                        transcript = item.get("fixed_transcript") or item.get("ocr") or item.get("asr") or ""
                        text_lookup[item["id"]] = f"{item['title']} {transcript}"
        print(f"  Text lookup: {len(text_lookup)} samples loaded for BERT fine-tuning")

    # Training loop
    best_val_f1 = 0.0
    best_test_f1 = 0.0
    best_state = None
    best_ema_state = None
    patience_counter = 0

    print(f"\nTraining config: lr={config.training.learning_rate}, "
          f"wd={config.training.weight_decay}, "
          f"dropout={config.model.dropout}, "
          f"batch={config.training.batch_size}, "
          f"epochs={config.training.epochs}, "
          f"threshold={config.training.decision_threshold}, "
          f"deep_fusion={args.deep_fusion}, "
          f"num_heads={args.num_heads}, "
          f"language_aware={args.language_aware}, "
          f"mixup_alpha={args.mixup_alpha}, "
          f"dann_alpha={args.dann_alpha}, "
          f"distill_alpha={args.distill_alpha}, "
          f"train_bert={args.train_bert}, "
          f"lora_rank={args.lora_rank}, "
          f"bert_pool={args.bert_pool}, "
          f"lang_loss_weights={language_loss_weights}, "
          f"select_metric={args.select_metric}")

    for epoch in range(config.training.epochs):
        # Scheduled en loss weight decay
        if args.en_weight_decay > 0 and language_loss_weights is not None:
            progress = epoch / max(1, config.training.epochs - 1)
            current_en = args.en_loss_weight + (args.en_weight_decay_target - args.en_loss_weight) * progress
            language_loss_weights = {"zh": args.zh_loss_weight, "en": current_en}

        # Train one epoch (EMA accumulates online model internally)
        train_loss, _, _ = train_epoch(
            model, train_loader, optimizer, criterion, device,
            ema_model=ema_state, ema_decay=ema_decay,
            mixup_alpha=args.mixup_alpha,
            distill_alpha=args.distill_alpha,
            distill_temperature=args.distill_temperature,
            language_loss_weights=language_loss_weights,
            text_lookup=text_lookup if args.train_bert else None,
            bert_tokenizer=bert_tokenizer if args.train_bert else None,
        )
        # Use EMA weights for evaluation if available
        if ema_state:
            saved_online = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_state)

        # Evaluate on validation set (if available) or test set
        if val_loader is not None:
            val_loss, val_labels, val_preds, val_langs = evaluate(
                model, val_loader, criterion, device,
                text_lookup=text_lookup if args.train_bert else None,
                bert_tokenizer=bert_tokenizer if args.train_bert else None,
            )
            val_metrics = compute_metrics(val_labels, val_preds, prefix="val_")
            add_per_dataset_metrics(val_metrics, val_labels, val_preds, val_langs, prefix="val_")
            if args.select_metric == "macro_f1":
                epoch_f1 = val_metrics.get("val_macro_f1", 0.0)
            elif args.select_metric == "min_sv_tt":
                epoch_f1 = min(val_metrics.get("val_sv_accuracy", 0), val_metrics.get("val_tt_accuracy", 0))
            elif args.select_metric == "weighted_sv_tt":
                epoch_f1 = 0.5 * val_metrics.get("val_sv_accuracy", 0) + 0.5 * val_metrics.get("val_tt_accuracy", 0)
            elif args.select_metric == "target_gap":
                epoch_f1 = min(
                    val_metrics.get("val_sv_accuracy", 0) / 0.86,
                    val_metrics.get("val_tt_accuracy", 0) / 0.84,
                )

            # Test set metrics for reporting (\u00a78 per-dataset metrics)
            test_loss, test_labels, test_preds, test_langs = evaluate(
                model, test_loader, criterion, device,
                text_lookup=text_lookup if args.train_bert else None,
                bert_tokenizer=bert_tokenizer if args.train_bert else None,
            )
            test_metrics = compute_metrics(test_labels, test_preds, prefix="test_")
            add_per_dataset_metrics(test_metrics, test_labels, test_preds, test_langs, prefix="test_")

            sv_acc = val_metrics.get('val_sv_accuracy', -1)
            tt_acc = val_metrics.get('val_tt_accuracy', -1)
            print(f"Epoch {epoch+1:2d}/{config.training.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Sel({args.select_metric}): {epoch_f1:.4f} | "
                  f"SV_acc={sv_acc:.4f} TT_acc={tt_acc:.4f} | "
                  f"Test Acc: {test_metrics['test_accuracy']:.4f} | "
                  f"SV F1: {test_metrics.get('test_sv_macro_f1', -1):.4f} "
                  f"TT F1: {test_metrics.get('test_tt_macro_f1', -1):.4f}")
        else:
            # No validation set \u2014 use test set for early stopping
            test_loss, test_labels, test_preds, test_langs = evaluate(
                model, test_loader, criterion, device,
                text_lookup=text_lookup if args.train_bert else None,
                bert_tokenizer=bert_tokenizer if args.train_bert else None,
            )
            test_metrics = compute_metrics(test_labels, test_preds, prefix="test_")
            add_per_dataset_metrics(test_metrics, test_labels, test_preds, test_langs, prefix="test_")

            if args.select_metric == "macro_f1":
                epoch_f1 = test_metrics.get("test_macro_f1", 0.0)
            elif args.select_metric == "min_sv_tt":
                epoch_f1 = min(test_metrics.get("test_sv_accuracy", 0), test_metrics.get("test_tt_accuracy", 0))
            elif args.select_metric == "weighted_sv_tt":
                epoch_f1 = 0.5 * test_metrics.get("test_sv_accuracy", 0) + 0.5 * test_metrics.get("test_tt_accuracy", 0)
            elif args.select_metric == "target_gap":
                epoch_f1 = min(
                    test_metrics.get("test_sv_accuracy", 0) / 0.86,
                    test_metrics.get("test_tt_accuracy", 0) / 0.84,
                )

            print(f"Epoch {epoch+1:2d}/{config.training.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_metrics['test_accuracy']:.4f} | "
                  f"Macro F1: {epoch_f1:.4f} "
                  f"(SV: {test_metrics.get('test_sv_macro_f1', -1):.4f}, "
                  f"TT: {test_metrics.get('test_tt_macro_f1', -1):.4f})")

        # Early stopping (based on validation F1 if available, else test F1)
        if epoch_f1 > (best_val_f1 if val_loader is not None else best_test_f1):
            if val_loader is not None:
                best_val_f1 = epoch_f1
            else:
                best_test_f1 = epoch_f1
            if ema_state:
                best_ema_state = {k: v.clone() for k, v in ema_state.items()}
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Restore online weights for next epoch if EMA was used
                if ema_state:
                    model.load_state_dict(saved_online)
                break

        # Restore online weights for next epoch's training
        if ema_state:
            model.load_state_dict(saved_online)

        # Step learning rate scheduler (after optimizer update for each epoch)
        if scheduler is not None:
            scheduler.step()

    # Restore best model (EMA state if available, else online state)
    if best_ema_state is not None:
        model.load_state_dict(best_ema_state)
        print("  Restored best EMA state")
    elif best_state is not None:
        model.load_state_dict(best_state)

    tuned_thresholds = None
    if args.tune_threshold and val_loader is not None:
        _, val_labels, _, val_langs, val_probs = evaluate(
            model, val_loader, criterion, device, return_probs=True,
            text_lookup=text_lookup if args.train_bert else None,
            bert_tokenizer=bert_tokenizer if args.train_bert else None,
        )
        tuned_thresholds, threshold_scores = tune_thresholds(
            val_labels,
            val_probs,
            val_langs,
            metric=args.threshold_metric,
            per_lang=args.per_lang_threshold,
        )
        print(f"  Tuned thresholds ({args.threshold_metric}): "
              f"{tuned_thresholds} scores={threshold_scores}")

    # Final evaluation on test set
    test_loss, test_labels, test_preds, test_langs, test_probs = evaluate(
        model, test_loader, criterion, device,
        thresholds=tuned_thresholds,
        return_probs=True,
        text_lookup=text_lookup if args.train_bert else None,
        bert_tokenizer=bert_tokenizer if args.train_bert else None,
    )
    final_metrics = compute_metrics(test_labels, test_preds, prefix="final_test_")
    add_per_dataset_metrics(final_metrics, test_labels, test_preds, test_langs, prefix="final_test_")
    if tuned_thresholds is not None:
        for key, value in tuned_thresholds.items():
            final_metrics[f"final_threshold_{key}"] = float(value)

    print("\n========== Final Results ==========")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    model_path = output_dir / "model.pt"
    if args.no_save_model:
        print("  Model save skipped (--no-save-model)")
    else:
        try:
            state_to_save = (
                compact_checkpoint_state(model)
                if args.compact_checkpoint else
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            torch.save(state_to_save, model_path)
            with open(output_dir / "checkpoint_meta.json", "w") as f:
                json.dump({
                    "compact_checkpoint": bool(args.compact_checkpoint),
                    "train_bert": bool(args.train_bert),
                    "lora_rank": int(args.lora_rank),
                    "bert_pool": args.bert_pool,
                    "init_checkpoint": args.init_checkpoint,
                    "freeze_bert_after_load": bool(args.freeze_bert_after_load),
                }, f, indent=2)
        except (OSError, RuntimeError) as exc:
            print(f"  WARNING: failed to save model checkpoint to {model_path}: {exc}")
            try:
                if model_path.exists():
                    model_path.unlink()
            except OSError:
                pass

    with open(output_dir / "predictions.jsonl", "w") as f:
        for label, pred, lang, prob in zip(test_labels, test_preds, test_langs, test_probs):
            f.write(json.dumps({
                "label": int(label),
                "pred": int(pred),
                "prob": float(prob),
                "lang": lang,
            }) + "\n")

    print(f"\nResults saved to {output_dir}")
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train from cached features (Stage 2)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio (\u00a77.4.2); default 0.2. "
                             "Set to 0.0 to use test set for early stopping (not recommended).")
    parser.add_argument("--ablate-module1", action="store_true",
                        help="Zero Module 1 output (FINT_FACT=0) for ablation (\u00a77.4.6 A1)")
    parser.add_argument("--ablate-module2", action="store_true",
                        help="Zero Module 2 output (FEXT_FACT=0) for ablation (\u00a77.4.6 A2)")
    parser.add_argument("--ablate-module3", action="store_true",
                        help="Zero Module 3 output (FSTYLE=0) for ablation (\u00a77.4.6 A3)")
    parser.add_argument("--ablate-text-aux", action="store_true",
                        help="Zero text_aux signal for ablation (\u00a77.4.6 A4)")
    parser.add_argument("--entity-conditioned", action="store_true",
                        help="Use EntityConditionedFactBranchMLP (entity stats \u2192 branch gating)")
    parser.add_argument("--ablate-entity-gating", action="store_true",
                        help="Disable ES gating in EntityConditionedFactBranchMLP (uniform weights)")
    # Level 3 simple replacement controls
    parser.add_argument("--simple-module1", action="store_true",
                        help="Replace Module 1 with simple averaging (C1)")
    parser.add_argument("--simple-module2", action="store_true",
                        help="Replace Module 2 with branch averaging (C2)")
    parser.add_argument("--simple-module3", action="store_true",
                        help="Replace Module 3 with single linear layer (C3)")
    parser.add_argument("--simple-fusion", action="store_true",
                        help="Replace two-stage fusion with concat-all classifier (C4)")
    # Level 3 revised: weak (weaken, not replace) controls
    parser.add_argument("--weak-module1", action="store_true",
                        help="C1 revised: reduce co-attention heads 4\u21921 (weaken cross-modal capacity)")
    parser.add_argument("--no-co-attn", action="store_true",
                        help="C1 v2: remove co-attention entirely, concat projections")
    parser.add_argument("--weak-fusion", action="store_true",
                        help="C4 revised: reduce fusion hidden_dim 256\u219216 (weaken fusion capacity)")
    # Level 5 data ablation controls
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of training data to use (0.0-1.0, default 1.0)")
    parser.add_argument("--test-lang", type=str, default=None,
                        choices=["all", "zh", "en"],
                        help="Override lang filter for test set only (default: same as --lang-filter)")
    # Hyperparameter overrides (optional; override config values if provided)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override weight decay")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override dropout rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--module-output-dim", type=int, default=None,
                        help="Override module output dimension")
    parser.add_argument("--mlp2-hidden", type=int, default=None,
                        help="Override Module 2 MLP hidden dimension")
    parser.add_argument("--text-aux-dim", type=int, default=None,
                        help="Override text_aux compression dimension")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early stopping patience")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Override cached feature directory")
    parser.add_argument("--lang-filter", type=str, default="all",
                        choices=["all", "zh", "en"],
                        help="Filter by language: zh (SV), en (TT), or all")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"],
                        help="Optimizer: adamw (default) or sgd (momentum=0.9, nesterov)")
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "cosine", "onecycle"],
                        help="LR scheduler: none, cosine, or onecycle (OneCycleLR)")
    parser.add_argument("--entity-stats-dim", type=int, default=0,
                        help="Compress 35-dim entity stats to this dim (0=disabled, default)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--val-seed", type=int, default=None,
                        help="Validation split seed (None = use training seed). "
                             "Fixing this reduces seed sensitivity.")
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bce", "focal"],
                        help="Loss function: bce (default) or focal")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal Loss alpha (class balance, default 0.25)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal Loss gamma (focus on hard examples, default 2.0)")
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help="Positive class weight for BCE (0=disabled, default). "
                             "E.g., use (neg_count/pos_count) to balance classes.")
    # Round 10: Training enhancements
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Fraction of total steps for linear LR warmup (0=disabled)")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="EMA decay rate (e.g., 0.999). 0=disabled.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for BCE (0=disabled, 0.1 typical). "
                             "Smooths hard 0/1 targets to alpha/1-alpha.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Max gradient norm for clipping (0=disabled)")
    # Round 11: Deep fusion + multi-head + mixup
    parser.add_argument("--deep-fusion", action="store_true",
                        help="Use DeepResidualFusion + MultiHeadClassifier")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="Number of classifier heads (1=standard single head)")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mixup alpha for Beta distribution (0=disabled, 0.5 typical)")
    # Round 12: single-model distillation and validation threshold tuning
    parser.add_argument("--teacher-probs", type=str, default=None,
                        help="JSONL teacher probabilities aligned by sample id or train row order")
    parser.add_argument("--distill-alpha", type=float, default=0.0,
                        help="Weight for teacher soft-label BCE loss (0=disabled)")
    parser.add_argument("--distill-temperature", type=float, default=1.0,
                        help="Temperature applied to student logits in distillation loss")
    parser.add_argument("--zh-loss-weight", type=float, default=1.0,
                        help="Per-sample loss multiplier for zh/FakeSV samples")
    parser.add_argument("--en-loss-weight", type=float, default=1.0,
                        help="Per-sample loss multiplier for en/FakeTT samples")
    parser.add_argument("--en-weight-decay", type=float, default=0.0,
                        help="Anneal en_loss_weight linearly to --en-weight-decay-target over training (0=no decay)")
    parser.add_argument("--en-weight-decay-target", type=float, default=1.0,
                        help="Target en_loss_weight at end of training when --en-weight-decay > 0")
    parser.add_argument("--en-oversample", type=int, default=1,
                        help="Oversample English samples by this factor (1=disabled, 2=2x, etc.)")
    parser.add_argument("--select-metric", type=str, default="macro_f1",
                        choices=["macro_f1", "min_sv_tt", "weighted_sv_tt", "target_gap"],
                        help="Validation metric for checkpoint selection and early stopping")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Tune decision threshold on validation predictions before final test eval")
    parser.add_argument("--per-lang-threshold", action="store_true",
                        help="Tune separate validation thresholds for zh/FakeSV and en/FakeTT")
    parser.add_argument("--threshold-metric", type=str, default="accuracy",
                        choices=["accuracy", "macro_f1"],
                        help="Validation metric for threshold tuning")
    parser.add_argument("--language-aware", action="store_true",
                        help="Append a learned zh/en language embedding to fusion features")
    parser.add_argument("--lang-emb-dim", type=int, default=8,
                        help="Language embedding dimension when --language-aware is enabled")
    parser.add_argument("--per-lang-classifier", action="store_true",
                        help="Use separate classification heads for zh and en")
    # Round 13: Trainable BERT for M3
    parser.add_argument("--train-bert", action="store_true",
                        help="Fine-tune BERT jointly during Stage 2 training")
    parser.add_argument("--bert-tune-layers", type=int, default=4,
                        help="Number of BERT top layers to fine-tune (0=all, default=4)")
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank for parameter-efficient BERT fine-tuning (0=disabled, 4..16 typical)")
    parser.add_argument("--bert-pool", type=str, default="cls",
                        choices=["cls", "mean"],
                        help="BERT pooling strategy: cls (CLS token) or mean (masked mean)")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Warm-start model weights from a checkpoint before training")
    parser.add_argument("--init-checkpoint-strict", action="store_true",
                        help="Require all checkpoint keys to match the current model")
    parser.add_argument("--dann-alpha", type=float, default=0.0,
                        help="Gradient reversal scaling for domain-adversarial training (0=disabled). "
                             "Encourages language-invariant features by reversing gradients from a "
                             "language discriminator. Typical range: 0.01-1.0.")
    parser.add_argument("--fact-fusion-extra", action="store_true",
                        help="Add extra Dropout+Linear layer in FactFusionMLP for more capacity")
    parser.add_argument("--freeze-bert-after-load", action="store_true",
                        help="Freeze BERT parameters after loading --init-checkpoint")
    parser.add_argument("--compact-checkpoint", action="store_true",
                        help="When saving trainable-BERT models, omit frozen BERT base tensors and keep LoRA/non-BERT weights")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Save metrics/predictions but skip model.pt")
    args = parser.parse_args()
    main(args)
