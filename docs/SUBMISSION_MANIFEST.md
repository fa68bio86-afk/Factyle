# Anonymous Submission Manifest

Included:

- `src/factyle/`: core package source code;
- `configs/`: default and small validation configs;
- `scripts/build_full_feature_cache.py`: Stage 1 feature cache builder;
- `scripts/train_from_cache.py`: Stage 2 training entry point;
- `scripts/verify_cache.py`: JSONL cache validation helper;
- `scripts/validate_modules.py`: lightweight component checks;
- selected evaluation and ablation helper scripts;
- `third_party/ImageBind/`: vendored ImageBind implementation and license;
- `README.md`, `docs/REPRODUCE.md`, and setup notes.

Excluded:

- datasets;
- model checkpoints;
- cached features;
- predictions and logs;
- local backup scripts;
- hyperparameter search round scripts;
- API credentials;
- local absolute paths;
- development git history.

Before upload, run local scans for:

- CJK characters;
- former project names;
- local absolute paths;
- API key prefixes or credential-looking strings.

The anonymous package should return no sensitive hits.
