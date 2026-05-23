"""Evaluate a trained checkpoint with an ablation flag override.

Usage:
  python scripts/eval_ablation_checkpoint.py \
      --checkpoint outputs/ablation/entity_conditioned/model.pt \
      --config configs/default.yaml \
      --output-dir outputs/ablation/entity_conditioned_ablated \
      --override ablate_entity_gating

This loads a trained model, enables the specified ablation override,
re-evaluates on both SV and TT, and saves metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from factyle.models.torch_models import FactStyleFusionClassifier
from factyle.utils.config import load_config
from train_from_cache import (
    CachedFeatureDataset, FeatureNormalizer, NormalizedDataset,
    collate_fn, evaluate, compute_metrics, add_per_dataset_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with ablation override")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model.pt checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (used for data paths only)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to save ablation metrics")
    parser.add_argument("--override", type=str, default="ablate_entity_gating",
                        choices=["ablate_entity_gating", "ablate_module1",
                                 "ablate_module2", "ablate_module3", "ablate_text_aux",
                                 "simple_module1", "simple_module2", "simple_module3",
                                 "simple_fusion", "no_co_attn", "weak_fusion"],
                        help="Ablation flag to enable for re-evaluation")
    parser.add_argument("--lang-filter", type=str, default="all",
                        choices=["all", "zh", "en"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config and checkpoint
    config = load_config(args.config)
    print(f"Config: {config.name}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Infer model params from checkpoint tensor shapes
    # We need: module_output_dim, mlp2_hidden, text_aux_dim, etc.
    # These are stored as model attributes \u2014 load from saved state keys
    print(f"Checkpoint loaded. State keys: {[k for k in checkpoint.keys() if 'weight' in k][:5]}...")
    print(f"  Total keys: {len(checkpoint)}")

    # Reconstruct model with matching params
    # The checkpoint was saved with specific dimensions \u2014 infer them
    m2_weight_shape = None
    for k, v in checkpoint.items():
        if "module2" in k and "weight" in k and v.dim() == 2:
            m2_weight_shape = v.shape
            break

    # Infer mlp2_hidden and module_output_dim from state dict
    # FactBranchMLP: net.0.weight (5376, hidden) \u2192 net.3.weight (hidden, output_dim)
    # EntityConditionedFactBranchMLP: branch_encoder.0.weight (768, hidden)
    #                                   output_proj.weight (hidden, output_dim)
    mlp2_hidden = config.model.mlp2_hidden
    module_output_dim = config.model.module_output_dim

    for k, v in checkpoint.items():
        if "module2_mlp.branch_encoder.0.weight" in k:
            mlp2_hidden = v.shape[0]
        elif "module2_mlp.branch_encoder.3.weight" in k:
            mlp2_hidden = v.shape[0]
        elif "module2_mlp.output_proj.weight" in k:
            module_output_dim = v.shape[0]
        elif "module2_mlp.net.0.weight" in k and k.count(".") == 3:
            mlp2_hidden = v.shape[0]
        elif "module2_mlp.net.3.weight" in k:
            module_output_dim = v.shape[0]

    # Check if it's entity_conditioned
    entity_conditioned = any("es_gate" in k for k in checkpoint)
    print(f"  Detected entity_conditioned={entity_conditioned}")
    print(f"  Detected mlp2_hidden={mlp2_hidden}, module_output_dim={module_output_dim}")

    # Build model
    model = FactStyleFusionClassifier(
        module_output_dim=module_output_dim,
        mlp2_hidden=mlp2_hidden,
        text_aux_dim=config.model.text_aux_dim,
        text_aux_input_dim=config.cache.hashing_dim + 12,
        dropout=config.model.dropout,
        entity_conditioned=entity_conditioned,
        # Apply the override
        ablate_entity_gating=(args.override == "ablate_entity_gating"),
        ablate_module1=(args.override == "ablate_module1"),
        ablate_module2=(args.override == "ablate_module2"),
        ablate_module3=(args.override == "ablate_module3"),
        ablate_text_aux=(args.override == "ablate_text_aux"),
        simple_module1=(args.override == "simple_module1"),
        simple_module2=(args.override == "simple_module2"),
        simple_module3=(args.override == "simple_module3"),
        simple_fusion=(args.override == "simple_fusion"),
        no_co_attn=(args.override == "no_co_attn"),
        weak_fusion=(args.override == "weak_fusion"),
    ).to(device)
    model.classifier_output_threshold = config.training.decision_threshold

    # Load checkpoint weights
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded. Active overrides: {args.override}={getattr(model, args.override, None)}")

    # Load test data
    cache_path_sv = Path(config.cache.cache_dir) / "entity_cache.jsonl"
    cache_path_tt = Path(str(cache_path_sv).replace("FakeSVDataset", "FakeTTDataset"))

    # Use original test set
    if args.lang_filter == "zh" or args.lang_filter == "all":
        print(f"\nLoading SV test data...")
        sv_dataset = CachedFeatureDataset(str(cache_path_sv), lang_filter="zh")
        # Use validation split to get test set
        from train_from_cache import create_validation_split
        _, sv_test = create_validation_split(sv_dataset, val_split=0.2, seed=42)
        print(f"  SV test samples: {len(sv_test)}")

        normalizer = FeatureNormalizer()
        normalizer_features = CachedFeatureDataset(str(cache_path_sv), lang_filter="zh")
        normalizer.fit(normalizer_features)
        sv_test_loader = DataLoader(
            NormalizedDataset(sv_test, normalizer),
            batch_size=config.training.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2
        )

        # Evaluate on SV
        sv_loss, sv_labels, sv_preds, sv_langs = evaluate(
            model, sv_test_loader, torch.nn.BCEWithLogitsLoss(), device
        )
        sv_metrics = compute_metrics(sv_labels, sv_preds, prefix="final_test_sv_")
        print(f"\n  SV Results (override={args.override}):")
        for k, v in sv_metrics.items():
            print(f"    {k}: {v:.4f}")

    if args.lang_filter == "en" or args.lang_filter == "all":
        print(f"\nLoading TT test data...")
        tt_dataset = CachedFeatureDataset(str(cache_path_tt), lang_filter="en")
        _, tt_test = create_validation_split(tt_dataset, val_split=0.2, seed=42)
        print(f"  TT test samples: {len(tt_test)}")

        normalizer_tt = FeatureNormalizer()
        normalizer_tt_feat = CachedFeatureDataset(str(cache_path_tt), lang_filter="en")
        normalizer_tt.fit(normalizer_tt_feat)
        tt_test_loader = DataLoader(
            NormalizedDataset(tt_test, normalizer_tt),
            batch_size=config.training.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2
        )

        # Evaluate on TT
        tt_loss, tt_labels, tt_preds, tt_langs = evaluate(
            model, tt_test_loader, torch.nn.BCEWithLogitsLoss(), device
        )
        tt_metrics = compute_metrics(tt_labels, tt_preds, prefix="final_test_tt_")
        print(f"\n  TT Results (override={args.override}):")
        for k, v in tt_metrics.items():
            print(f"    {k}: {v:.4f}")

    # Combine and save
    all_metrics = {}
    if args.lang_filter in ["zh", "all"]:
        all_metrics.update(sv_metrics)
    if args.lang_filter in ["en", "all"]:
        all_metrics.update(tt_metrics)
    all_metrics["override"] = args.override

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
