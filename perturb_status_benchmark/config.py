"""Configuration loading, path resolution and deterministic fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_METHODS = {"mixscape", "mixscale", "ps", "target_expression"}


def load_config(path: Path) -> dict[str, Any]:
    config = _read_with_extends(path.resolve(), set())
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a YAML mapping")
    for key in ("protocol", "project_root", "datasets", "cohort", "resources"):
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    project_root = Path(config["project_root"])
    if not project_root.is_absolute():
        project_root = (path.resolve().parent / project_root).resolve()
    else:
        project_root = project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")
    config["project_root"] = project_root
    config["direct_expression"]["cell_table"] = resolve_path(project_root, config["direct_expression"]["cell_table"])
    for key in ("output_root", "work_root", "log_root"):
        config[key] = resolve_path(project_root, config[key])
    if not config["datasets"]:
        raise ValueError("At least one dataset is required")
    for dataset_id, dataset in config["datasets"].items():
        missing = REQUIRED_METHODS - set(dataset.get("methods", {}))
        if missing:
            raise ValueError(f"{dataset_id} is missing methods: {sorted(missing)}")
        dataset["h5ad"] = resolve_path(project_root, dataset["h5ad"])
        expression = dataset.get("expression", {})
        if expression.get("target_feature_map"):
            expression["target_feature_map"] = resolve_path(project_root, expression["target_feature_map"])
        for method in dataset["methods"].values():
            method["path"] = resolve_path(project_root, method["path"])
    config["config_path"] = path.resolve()
    config["fingerprint"] = fingerprint(config)
    return config


def _read_with_extends(path: Path, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"Circular config inheritance involving {path}")
    with path.open(encoding="utf-8") as handle:
        current = yaml.safe_load(handle)
    if not isinstance(current, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    parent_value = current.pop("extends", None)
    if parent_value is None:
        return current
    parent = Path(parent_value)
    if not parent.is_absolute():
        parent = (path.parent / parent).resolve()
    base = _read_with_extends(parent, {*seen, path})
    return _deep_merge(base, current)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def fingerprint(config: dict[str, Any]) -> str:
    excluded = {"fingerprint", "config_path"}

    def serializable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in sorted(value.items()) if k not in excluded}
        if isinstance(value, list):
            return [serializable(v) for v in value]
        return value

    payload = json.dumps(serializable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
