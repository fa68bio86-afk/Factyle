"""Configuration management.

Reference: ARCHITECTURE.md \u00a77
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Module output dimensions
    module_output_dim: int = 256
    mlp2_hidden: int = 1024
    text_aux_dim: int = 64
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    batch_size: int = 32
    epochs: int = 12
    seed: int = 42
    decision_threshold: float = 0.5
    patience: int = 5  # early stopping patience


@dataclass
class DataConfig:
    fake_sv_root: str = os.environ.get("FACTYLE_FAKE_SV_ROOT", "data/FakeSVDataset")
    fake_tt_root: str = os.environ.get("FACTYLE_FAKE_TT_ROOT", "data/FakeTTDataset")
    max_train_items: Optional[int] = None
    max_test_items: Optional[int] = None


@dataclass
class FeatureCacheConfig:
    cache_dir: str = "outputs/feature_cache"
    hashing_dim: int = 512
    num_video_frames: int = 12
    api_workers: int = 7


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_dir: str = "outputs/experiments/default"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _parse_config(raw)


def _parse_config(raw: dict) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.name = raw.get("name", cfg.name)
    cfg.output_dir = raw.get("output_dir", cfg.output_dir)

    if "model" in raw:
        m = raw["model"]
        cfg.model.module_output_dim = m.get("module_output_dim", cfg.model.module_output_dim)
        cfg.model.mlp2_hidden = m.get("mlp2_hidden", cfg.model.mlp2_hidden)
        cfg.model.text_aux_dim = m.get("text_aux_dim", cfg.model.text_aux_dim)
        cfg.model.dropout = m.get("dropout", cfg.model.dropout)

    if "training" in raw:
        t = raw["training"]
        cfg.training.learning_rate = t.get("learning_rate", cfg.training.learning_rate)
        cfg.training.weight_decay = t.get("weight_decay", cfg.training.weight_decay)
        cfg.training.batch_size = t.get("batch_size", cfg.training.batch_size)
        cfg.training.epochs = t.get("epochs", cfg.training.epochs)
        cfg.training.seed = t.get("seed", cfg.training.seed)
        cfg.training.decision_threshold = t.get("decision_threshold", cfg.training.decision_threshold)
        cfg.training.patience = t.get("patience", cfg.training.patience)

    if "data" in raw:
        d = raw["data"]
        cfg.data.fake_sv_root = d.get("fake_sv_root", cfg.data.fake_sv_root)
        cfg.data.fake_tt_root = d.get("fake_tt_root", cfg.data.fake_tt_root)
        cfg.data.max_train_items = d.get("max_train_items", cfg.data.max_train_items)
        cfg.data.max_test_items = d.get("max_test_items", cfg.data.max_test_items)

    if "cache" in raw:
        c = raw["cache"]
        cfg.cache.cache_dir = c.get("cache_dir", cfg.cache.cache_dir)
        cfg.cache.hashing_dim = c.get("hashing_dim", cfg.cache.hashing_dim)
        cfg.cache.num_video_frames = c.get("num_video_frames", cfg.cache.num_video_frames)
        cfg.cache.api_workers = c.get("api_workers", cfg.cache.api_workers)

    return cfg
