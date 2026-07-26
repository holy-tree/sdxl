import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(cfg)}")
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_output_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    out_dir = Path(cfg.get("output_dir", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out_dir)
    cfg["logging_dir"] = str(out_dir / cfg.get("logging_dir", "logs"))
    Path(cfg["logging_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def to_namespace(cfg: Dict[str, Any]):
    import argparse

    return argparse.Namespace(**cfg)
