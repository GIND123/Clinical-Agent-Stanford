"""YAML config loading with a simple `base:` include mechanism.

A config file may name `base: <path>` (relative to itself); the base is loaded first
and the child is deep-merged on top. Paths ending in `_path` / `_file` / `_dir` are
left as written (relative to the working directory), which keeps CLI usage predictable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` overrides (value parsed as YAML)."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg
