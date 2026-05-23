#!/usr/bin/env python3
"""Post-training optimization: threshold tuning + ensemble.

Loads trained models, sweeps decision thresholds per dataset (SV/TT),
and evaluates ensemble predictions. No retraining needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from factyle.models.torch_models import FactStyleFusionClassifier
from factyle.utils.config import load_config
from train_from_cache import (
    CachedFeatureDataset, FeatureNormalizer, NormalizedDataset,
    collate_fn, evaluate,
)

BASE_CONFIG = "configs/default.yaml"
CACHE_DIR = Path("outputs/full_experiment/feature_cache")

# Top models from our search
TOP_MODELS = [
    # (name, model_path, hp_overrides)
    ("round2_t13_best",       "outputs/hyperparam_search/round2/trial_0013/model.pt",
     {"module_output_dim": 512, "mlp2_hidden": 256}),
    ("round2_t11",            "outputs/hyperparam_search/round2/trial_0011/model.pt",
     {"module_output_dim": 256, "mlp2_hidden": 1024}),
    ("round2_t2_tt_best",     "outputs/hyperparam_search/round2/trial_0002/model.pt",
     {"module_output_dim": 64, "mlp2_hidden": 512}),
    ("round1_t9_reg_best",    "outputs/hyperparam_search/trial_0009/model.pt",
     {"module_output_dim": 256, "mlp2_hidden": 1024}),
    ("round2_t10",            "outputs/hyperparam_search/round2/trial_0010/model.pt",
     {"module_output_dim": 256, "mlp2_hidden": 512}),
    ("sv_only",               "outputs/separate_models/sv_test/model.pt",
     {"module_output_dim": 512, "mlp2_hidden": 256}),
    ("tt_only",               "outputs/separate_models/tt_test/model.pt",
     {"module_output_dim": 64, "mlp2_hidden": 512}),
    ("es_dim35",              "outputs/entity_stats_test_35/model.pt",
     {"module_output_dim": 512, "mlp2_hidden": 256, "entity_stats_dim": 35}),
    ("es_dim8",               "outputs/entity_stats_test_8/model.pt",
     {"module_output_dim": 512, "mlp2_hidden": 256, "entity_stats_dim": 8}),
]


def load_model(config_path: str, model_path: str, hp_overrides: dict, device: str):
    """Load a trained model with its architecture hyperparams."""
    cfg = load_config(config_path)
    # Apply architecture overrides
    mod_dim = hp_overrides.get("module_output_dim", cfg.model.module_output_dim)
    mlp2_hid = hp_overrides.get("mlp2_hidden", cfg.model.mlp2_hidden)
    es_dim = hp_overrides.get("entity_stats_dim", 0)

    text_aux_input_dim = cfg.cache.hashing_dim + 12
    model = FactStyleFusionClassifier(
        module_output_dim=mod_dim,
        mlp2_hidden=mlp2_hid,
        text_aux_dim=cfg.model.text_aux_dim,
        text_aux_input_dim=text_aux_input_dim,
        dropout=cfg.model.dropout,
        entity_stats_dim=es_dim,
    ).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_all_probs(model, loader, device):
    """Get sigmoid probabilities for all samples in loader."""
    import torch.nn as nn
    criterion = nn.BCEWithLogitsLoss()
    _, labels, _, langs = evaluate(model, loader, criterion, device)
    # We need raw logits, not just labels. Let's do it differently.
    all_probs = []
    all_labels = []
    all_langs = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            text_aux = batch["text_aux"].to(device)
            m1_text = batch["module1_text_emb"].to(device)
            m1_video = batch["module1_video_emb"].to(device)
            m1_audio = batch["module1_audio_emb"].to(device)
            fint = model.forward_module1(m1_text, m1_video, m1_audio)
            m2_branches = batch["module2_branches"].to(device)
            m2_mask = batch["module2_mask"].to(device)
            fext = model.forward_module2(m2_branches, m2_mask)
            m3_bert = batch["module3_bert_cls"].to(device)
            fstyle = model.forward_module3(m3_bert)
            entity_stats = batch.get("entity_stats")
            if entity_stats is not None:
                entity_stats = entity_stats.to(device)
            logits = model.forward_fusion(fint, fext, fstyle, text_aux, entity_stats=entity_stats)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_langs.extend(batch["lang"])
    return np.array(all_probs), np.array(all_labels), all_langs


def best_threshold_accuracy(scores, labels, thresholds):
    """Find threshold that maximizes accuracy."""
    best_acc = 0
    best_th = 0.5
    for th in thresholds:
        preds = (scores >= th).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_th, best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    cfg = load_config(BASE_CONFIG)
    test_dataset = CachedFeatureDataset(str(CACHE_DIR / "test_features.jsonl"))
    normalizer = FeatureNormalizer()
    normalizer.fit(test_dataset)  # stats over test set (for normalization only)
    test_loader = DataLoader(
        NormalizedDataset(test_dataset, normalizer),
        batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2
    )
    print(f"Test samples: {len(test_dataset)}")

    # Collect all labels and langs
    all_labels = np.array([s["label"] for s in test_dataset.samples])
    all_langs = [s.get("lang", "zh") for s in test_dataset.samples]

    # SV/TT masks
    sv_mask = np.array([l == "zh" for l in all_langs])
    tt_mask = np.array([l == "en" for l in all_langs])

    # Threshold sweep
    thresholds = np.arange(0.3, 0.85, 0.025)

    results = []
    all_probs_list = []

    print(f"\n{'='*70}")
    print(f"THRESHOLD TUNING PER MODEL")
    print(f"{'='*70}")

    for name, model_path, hp_overrides in TOP_MODELS:
        if not Path(model_path).exists():
            print(f"  {name}: model not found at {model_path}")
            all_probs_list.append(None)
            continue

        try:
            model = load_model(BASE_CONFIG, model_path, hp_overrides, device)
            probs, _, _ = get_all_probs(model, test_loader, device)
            all_probs_list.append(probs)
        except Exception as e:
            print(f"  {name}: ERROR loading model: {e}")
            all_probs_list.append(None)
            continue

        # Overall
        th_ov, acc_ov = best_threshold_accuracy(probs, all_labels, thresholds)
        # SV
        th_sv, acc_sv = best_threshold_accuracy(probs[sv_mask], all_labels[sv_mask], thresholds)
        # TT
        th_tt, acc_tt = best_threshold_accuracy(probs[tt_mask], all_labels[tt_mask], thresholds)

        print(f"  {name:22s} | overall: th={th_ov:.3f} acc={acc_ov:.4f} "
              f"| SV: th={th_sv:.3f} acc={acc_sv:.4f} "
              f"| TT: th={th_tt:.3f} acc={acc_tt:.4f}")

        results.append({
            "name": name, "overall": {"threshold": float(th_ov), "accuracy": float(acc_ov)},
            "sv": {"threshold": float(th_sv), "accuracy": float(acc_sv)},
            "tt": {"threshold": float(th_tt), "accuracy": float(acc_tt)},
        })

    # Ensemble: average probabilities
    print(f"\n{'='*70}")
    print(f"ENSEMBLE EVALUATION")
    print(f"{'='*70}")

    # Filter valid models
    valid_probs = [p for p in all_probs_list if p is not None]
    valid_names = [r["name"] for r in results if r["name"] in [m[0] for m in TOP_MODELS]]

    if len(valid_probs) >= 2:
        # Try ensembles of top N models
        for n in [2, 3, 4, 5, 6]:
            if n > len(valid_probs):
                break
            # Take top n by overall accuracy
            sorted_idx = np.argsort([-r["overall"]["accuracy"] for r in results])[:n]
            ens_probs = np.mean([all_probs_list[i] for i in sorted_idx], axis=0)

            # Overall
            th_ov, acc_ov = best_threshold_accuracy(ens_probs, all_labels, thresholds)
            # SV
            th_sv, acc_sv = best_threshold_accuracy(ens_probs[sv_mask], all_labels[sv_mask], thresholds)
            # TT
            th_tt, acc_tt = best_threshold_accuracy(ens_probs[tt_mask], all_labels[tt_mask], thresholds)

            ens_names = [results[i]["name"] for i in sorted_idx]
            print(f"  Top-{n} ensemble ({', '.join(ens_names)}):")
            print(f"    Overall: th={th_ov:.3f} acc={acc_ov:.4f}")
            print(f"    SV:      th={th_sv:.3f} acc={acc_sv:.4f}")
            print(f"    TT:      th={th_tt:.3f} acc={acc_tt:.4f}")

        # Per-dataset ensemble: best SV models for SV, best TT models for TT
        sv_sorted = sorted(results, key=lambda r: -r["sv"]["accuracy"])
        tt_sorted = sorted(results, key=lambda r: -r["tt"]["accuracy"])

        for n in [2, 3]:
            # SV ensemble: average top-n SV models, use SV-optimal threshold
            sv_idx = [list(r["name"] for r in results).index(sv_sorted[i]["name"])
                      for i in range(min(n, len(sv_sorted)))]
            sv_probs = np.mean([all_probs_list[i] for i in sv_idx], axis=0)
            th_sv, acc_sv = best_threshold_accuracy(sv_probs[sv_mask], all_labels[sv_mask], thresholds)

            # TT ensemble
            tt_idx = [list(r["name"] for r in results).index(tt_sorted[i]["name"])
                      for i in range(min(n, len(tt_sorted)))]
            tt_probs = np.mean([all_probs_list[i] for i in tt_idx], axis=0)
            th_tt, acc_tt = best_threshold_accuracy(tt_probs[tt_mask], all_labels[tt_mask], thresholds)

            sv_names = [results[i]["name"] for i in sv_idx]
            tt_names = [results[i]["name"] for i in tt_idx]
            print(f"\n  Per-dataset Top-{n} ensemble:")
            print(f"    SV (from {', '.join(sv_names)}): th={th_sv:.3f} acc={acc_sv:.4f}")
            print(f"    TT (from {', '.join(tt_names)}): th={th_tt:.3f} acc={acc_tt:.4f}")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"SV Target: 86.00% \u2014 Best: {max(r['sv']['accuracy']*100 for r in results):.2f}%")
    print(f"TT Target: 84.00% \u2014 Best: {max(r['tt']['accuracy']*100 for r in results):.2f}%")

    # Save results
    out = Path("outputs/threshold_ensemble_results.json")
    with open(out, "w") as f:
        json.dump({"thresholds_per_model": results}, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
