# Experiment Setup

## Training Scope

Stage 1 is feature extraction only:

- ImageBind runs in evaluation mode to extract text, video, and audio embeddings.
- Qwen models are called through API endpoints for entity extraction, style rewriting, and style analysis.
- BERT-base-multilingual runs in evaluation mode to extract cached CLS features.

Stage 2 trains the Factyle classifier from cached features. Optional BERT-LoRA training is available through `--train-bert --lora-rank`.

## Main Components

| Component | Role | Training status |
| --- | --- | --- |
| ImageBind huge | Multimodal feature extraction | Frozen |
| Qwen3-8B | Entity extraction and style rewriting | API inference |
| Qwen3-32B | Style analysis | API inference |
| BERT-base-multilingual | Cached CLS features and optional LoRA | Frozen or LoRA |
| Factyle MLP modules | Fusion and classification | Trainable |

## Metrics

The training script reports accuracy, macro F1, fake precision, fake recall, fake F1, real precision, real recall, and real F1. Metrics are reported for the combined test set and separately for FakeSV (`zh`) and FakeTT (`en`) when both are present.

## Baseline Policy

The code is intended for within-project comparisons on the same processed data subset. Direct comparison with numbers from original dataset papers may be unfair if the sample filtering, feature extraction, or evaluation protocol differs.

Recommended comparisons:

- full model versus module ablations;
- simplified fusion variants;
- data fraction experiments;
- repeated seeds using the same cached features;
- rerun external baselines on the same processed subset when possible.

