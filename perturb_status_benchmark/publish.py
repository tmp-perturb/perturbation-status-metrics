"""Atomic publication of a globally validated benchmark run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import yaml


def publish(config: dict[str, Any], run_id: str, run_root: Path, required_stages: list[str]) -> Path:
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    incomplete = [stage for stage in required_stages if manifest.get("stages", {}).get(stage, {}).get("status") != "complete"]
    if incomplete:
        raise RuntimeError(f"publication blocked; incomplete stages: {incomplete}")
    destination = config["output_root"] / run_id
    if destination.exists():
        raise FileExistsError(f"published output is immutable and already exists: {destination}")
    temporary = destination.with_name(destination.name + ".publishing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    _copy(run_root / "stages" / "validate" / "harmonized", temporary / "harmonized")
    _copy(run_root / "stages" / "embedding" / "expression", temporary / "expression" / "embedding")
    _copy(run_root / "stages" / "pseudobulk" / "pseudobulk", temporary / "expression" / "pseudobulk")
    for stage in ["coverage", "target_expression", "effect", "reproducibility", "agreement", "resources"]:
        target = temporary / "metrics" / stage
        target.mkdir(parents=True, exist_ok=True)
        for source in (run_root / "stages" / stage).glob("*.tsv"):
            shutil.copy2(source, target / source.name)
    _copy(run_root / "stages" / "report" / "benchmark_summary.tsv", temporary / "tables" / "benchmark_summary.tsv")
    _copy(run_root / "stages" / "report" / "benchmark_report.md", temporary / "report" / "benchmark_report.md")
    _copy(run_root / "run_manifest.json", temporary / "manifest" / "run_manifest.json")
    frozen = temporary / "manifest" / "resolved_config.yaml"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(yaml.safe_dump(_serializable(config), sort_keys=False), encoding="utf-8")
    checksums = _checksums(temporary)
    (temporary / "manifest" / "sha256sums.tsv").write_text(
        "sha256\tpath\n" + "\n".join(f"{digest}\t{path}" for path, digest in checksums) + "\n",
        encoding="utf-8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    return destination


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items() if key != "fingerprint"}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _checksums(root: Path) -> list[tuple[str, str]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.relative_to(root) == Path("manifest/run_manifest.json"):
            # The runner finalizes publication status immediately after the atomic rename.
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        rows.append((str(path.relative_to(root)), digest.hexdigest()))
    return rows
