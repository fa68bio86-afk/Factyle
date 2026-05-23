#!/usr/bin/env python3
"""Build single-sample teacher probabilities from the v10 ensemble.

The output JSONL is aligned by sample id and row order, so it can be passed to
`scripts/train_from_cache.py --teacher-probs ...` for single-model distillation.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from factyle.models.torch_models import FactStyleFusionClassifier
from train_from_cache import (
    CachedFeatureDataset,
    FeatureNormalizer,
    NormalizedDataset,
    collate_fn,
)


V10_MODELS = [
    ("t13_s300", 512, 256, 64, 0, "outputs/hyperparam_search/round3/trial_0003_t13_arch_seed300/model.pt"),
    ("t13", 512, 256, 64, 0, "outputs/hyperparam_search/round2/trial_0013/model.pt"),
    ("t2_s200", 64, 512, 64, 0, "outputs/hyperparam_search/round3/trial_0012_t2_arch_seed200/model.pt"),
    ("t2", 64, 512, 64, 0, "outputs/hyperparam_search/round2/trial_0002/model.pt"),
    ("t11_s1000", 256, 1024, 64, 0, "outputs/hyperparam_search/round3/trial_0010_t11_arch_seed1000/model.pt"),
    ("t11", 256, 1024, 64, 0, "outputs/hyperparam_search/round2/trial_0011/model.pt"),
    ("es35", 512, 256, 64, 35, "outputs/entity_stats_test_35/model.pt"),
    ("t11_sgd", 256, 1024, 64, 0, "outputs/hyperparam_search/round5/trial_0014_t11_arch_sgd_wd0.0001/model.pt"),
    ("t13_es35passthrough", 512, 256, 64, 35, "outputs/hyperparam_search/round6/trial_0013_t13_es35/model.pt"),
    ("t2_long_cos", 64, 512, 64, 0, "outputs/hyperparam_search/round7/trial_0012_t2_long_cos/model.pt"),
    ("t11_focal_g2", 256, 1024, 64, 0, "outputs/hyperparam_search/round7/trial_0020_t11_focal_g2.0/model.pt"),
    ("t13_tad32", 512, 256, 32, 0, "outputs/hyperparam_search/round8/trial_0001_t13_tad32/model.pt"),
    ("t11_focal_a075", 256, 1024, 64, 0, "outputs/hyperparam_search/round8/trial_0005_t11_focal_a0.75/model.pt"),
    ("t13_es35_lr001", 512, 256, 64, 35, "outputs/hyperparam_search/round8/trial_0009_t13_es35_lr0.001/model.pt"),
    ("t2_cos_lr001", 64, 512, 64, 0, "outputs/hyperparam_search/round8/trial_0012_t2_cos_lr0.001/model.pt"),
    ("t11_focal_lr001", 256, 1024, 64, 0, "outputs/hyperparam_search/round8/trial_0015_t11_focal_lr0.001/model.pt"),
]


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_prob(raw_prob, threshold, temperature):
    centered = logit(raw_prob) - logit(threshold)
    return sigmoid(centered / max(temperature, 1e-6))


def load_dataset(cache_path, batch_size):
    ds = CachedFeatureDataset(cache_path)
    norm = FeatureNormalizer()
    norm.fit(ds)
    loader = DataLoader(
        NormalizedDataset(ds, norm),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    return ds, loader


def compute_model_probs(loader, device):
    probs_list = []
    names = []
    for name, mod_dim, mlp2_hid, text_aux_dim, es_dim, path in V10_MODELS:
        if not Path(path).exists():
            print(f"SKIP {name}: missing {path}")
            continue
        print(f"Loading {name}...", flush=True)
        model = FactStyleFusionClassifier(
            module_output_dim=mod_dim,
            mlp2_hidden=mlp2_hid,
            text_aux_dim=text_aux_dim,
            text_aux_input_dim=524,
            dropout=0.5,
            entity_stats_dim=es_dim,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()

        probs = []
        with torch.no_grad():
            for batch in loader:
                text_aux = batch["text_aux"].to(device)
                fint = model.forward_module1(
                    batch["module1_text_emb"].to(device),
                    batch["module1_video_emb"].to(device),
                    batch["module1_audio_emb"].to(device),
                )
                fext = model.forward_module2(
                    batch["module2_branches"].to(device),
                    batch["module2_mask"].to(device),
                )
                fstyle = model.forward_module3(batch["module3_bert_cls"].to(device))
                entity_stats = batch.get("entity_stats")
                if entity_stats is not None and es_dim > 0:
                    entity_stats = entity_stats.to(device)
                logits = model.forward_fusion(
                    fint, fext, fstyle, text_aux, entity_stats=entity_stats
                )
                probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())

        names.append(name)
        probs_list.append(np.asarray(probs, dtype=np.float32))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return names, np.asarray(probs_list, dtype=np.float32)


def write_teacher_probs(ds, names, probs, ensemble_result, output_path, temperature):
    sv_weights = np.asarray(
        [ensemble_result["best_sv"]["weights"][name] for name in names],
        dtype=np.float32,
    )
    tt_weights = np.asarray(
        [ensemble_result["best_tt"]["weights"][name] for name in names],
        dtype=np.float32,
    )
    sv_threshold = float(ensemble_result["best_sv"]["threshold"])
    tt_threshold = float(ensemble_result["best_tt"]["threshold"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_values = []
    calibrated_values = []
    with open(output_path, "w") as f:
        for idx, sample in enumerate(ds.samples):
            lang = sample.get("lang", "zh")
            if lang == "zh":
                raw_prob = float(sv_weights @ probs[:, idx])
                threshold = sv_threshold
            else:
                raw_prob = float(tt_weights @ probs[:, idx])
                threshold = tt_threshold
            teacher_prob = float(calibrate_prob(raw_prob, threshold, temperature))
            raw_values.append(raw_prob)
            calibrated_values.append(teacher_prob)
            f.write(json.dumps({
                "row": idx,
                "id": sample.get("id"),
                "label": int(sample["label"]),
                "lang": lang,
                "teacher_raw_prob": raw_prob,
                "teacher_threshold": threshold,
                "teacher_prob": teacher_prob,
                "teacher_pred": int(teacher_prob >= 0.5),
            }) + "\n")

    print(f"Wrote {len(ds)} teacher rows to {output_path}")
    print(f"  raw mean={np.mean(raw_values):.4f}, calibrated mean={np.mean(calibrated_values):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="outputs/full_experiment/feature_cache/train_features.jsonl")
    parser.add_argument("--test-cache", default="outputs/full_experiment/feature_cache/test_features.jsonl")
    parser.add_argument("--ensemble-json", default="outputs/final_ensemble_results_v10.json")
    parser.add_argument("--output-dir", default="outputs/teacher_probs/v10")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-temperature", type=float, default=1.0)
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=["train", "test"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ensemble_result = json.loads(Path(args.ensemble_json).read_text())
    output_dir = Path(args.output_dir)
    for split in args.splits:
        cache_path = args.train_cache if split == "train" else args.test_cache
        print(f"\n=== {split}: {cache_path} ===")
        ds, loader = load_dataset(cache_path, args.batch_size)
        names, probs = compute_model_probs(loader, device)
        write_teacher_probs(
            ds,
            names,
            probs,
            ensemble_result,
            output_dir / f"{split}_teacher_probs.jsonl",
            args.calibration_temperature,
        )


if __name__ == "__main__":
    main()
