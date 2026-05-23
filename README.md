# Factyle

Factyle is a fake news detection research prototype for multimodal short-video news. It combines three evidence streams:

- cross-modal consistency from ImageBind text, video, and audio embeddings;
- entity-level retrieval features from web search and LLM-based extraction;
- style-aware rewriting and analysis features.

The repository is organized for paper review and reproduction. It excludes datasets, API credentials, checkpoints, cached features, logs, and local machine paths.

## Repository Layout

```text
configs/                  YAML experiment configs
docs/                     Reproduction and submission notes
scripts/                  Stage 1 cache building, Stage 2 training, and evaluation helpers
src/factyle/              Core Python package
third_party/ImageBind/    Vendored ImageBind code required by Stage 1
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Additional local assets expected by Stage 1:

```text
models/imagebind/imagebind_huge.pth
models/bert-base-multilingual-cased/
```

The project expects the datasets in this layout unless paths are overridden in `configs/default.yaml`:

```text
data/FakeSVDataset/
  train_title_transcript.json
  test_title_transcript.json
  filtered_video/
  audio/

data/FakeTTDataset/
  train_title_transcript.json
  test_title_transcript.json
  video/
  keyframe_audio/
```

## API Credentials

Stage 1 uses external APIs for retrieval and LLM calls. Create a local `.env` file:

```bash
cp .env.example .env
```

Then set:

```text
BAIDU_SEARCH_API_KEY
QWEN_API_KEY
QWEN_API_BASE
```

Do not commit `.env`.

## Stage 1: Build Feature Cache

```bash
python scripts/build_full_feature_cache.py \
  --config configs/default.yaml \
  --split all \
  --device cuda
```

For a fast offline check of Module 3 without Qwen calls:

```bash
python scripts/build_full_feature_cache.py \
  --config configs/tiny_validate.yaml \
  --split all \
  --device cuda \
  --offline-m3
```

Verify generated cache files:

```bash
python scripts/verify_cache.py outputs/full_experiment/feature_cache
```

## Stage 2: Train From Cache

```bash
python scripts/train_from_cache.py \
  --config configs/default.yaml \
  --val-split 0.2 \
  --output-dir outputs/full_experiment
```

Example with BERT-LoRA and compact checkpoints:

```bash
python scripts/train_from_cache.py \
  --config configs/default.yaml \
  --val-split 0.2 \
  --train-bert \
  --lora-rank 16 \
  --bert-pool mean \
  --compact-checkpoint \
  --output-dir outputs/full_experiment_lora
```

## Ablations

`scripts/train_from_cache.py` includes the main ablation flags used in the paper:

```text
--ablate-module1
--ablate-module2
--ablate-module3
--ablate-text-aux
--simple-module1
--simple-module2
--simple-module3
--simple-fusion
--no-co-attn
--weak-fusion
--data-fraction
```

See `docs/REPRODUCE.md` for a reproducible command template.

## What Is Not Included

This anonymous package intentionally omits:

- raw datasets;
- cached JSONL features;
- trained checkpoints;
- prediction files and logs;
- API keys;
- local backup scripts;
- hyperparameter search round scripts;
- git history from the development workspace.

