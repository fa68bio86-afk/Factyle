# Reproduction Guide

This guide describes the minimal workflow for reproducing Factyle experiments from public code and local data.

## 1. Prepare Data and Models

Place datasets under:

```text
data/FakeSVDataset/
data/FakeTTDataset/
```

Place pretrained model assets under:

```text
models/imagebind/imagebind_huge.pth
models/bert-base-multilingual-cased/
```

If your paths differ, update `configs/default.yaml`.

## 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install any additional ImageBind runtime dependencies required by your local GPU and PyTorch setup.

## 3. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and set the Baidu Search and Qwen credentials. Stage 2 training from an existing cache does not require API calls.

## 4. Build Cached Features

```bash
python scripts/build_full_feature_cache.py \
  --config configs/default.yaml \
  --split all \
  --device cuda \
  --workers 8
```

The cache is written to the `cache.cache_dir` value in the config, by default:

```text
outputs/full_experiment/feature_cache/
```

Check the cache:

```bash
python scripts/verify_cache.py outputs/full_experiment/feature_cache
```

## 5. Train the Model

```bash
python scripts/train_from_cache.py \
  --config configs/default.yaml \
  --val-split 0.2 \
  --seed 42 \
  --output-dir outputs/full_experiment
```

The output directory contains:

```text
metrics.json
predictions.jsonl
model.pt
```

Use `--compact-checkpoint` to save only trainable tensors for BERT-LoRA runs.

## 6. Example Paper-Style Run

```bash
python scripts/train_from_cache.py \
  --config configs/default.yaml \
  --val-split 0.2 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --dropout 0.2 \
  --module-output-dim 256 \
  --mlp2-hidden 1024 \
  --epochs 40 \
  --patience 12 \
  --scheduler onecycle \
  --warmup-ratio 0.1 \
  --train-bert \
  --lora-rank 16 \
  --bert-pool mean \
  --tune-threshold \
  --per-lang-threshold \
  --compact-checkpoint \
  --fact-fusion-extra \
  --output-dir outputs/paper_run
```

## 7. Ablation Examples

Remove Module 1:

```bash
python scripts/train_from_cache.py \
  --config configs/default.yaml \
  --val-split 0.2 \
  --ablate-module1 \
  --output-dir outputs/ablation_no_module1
```

Train on a 25 percent data subset:

```bash
python scripts/train_from_cache.py \
  --config configs/default.yaml \
  --val-split 0.2 \
  --data-fraction 0.25 \
  --output-dir outputs/data_fraction_25
```

## 8. Notes on Reproducibility

Exact scores depend on the local data subset, API responses, pretrained model checkpoints, random seed, and GPU/PyTorch stack. For a strict reproduction of reported results, reuse the same cached feature files and the same training command.
