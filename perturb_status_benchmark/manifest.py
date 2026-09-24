"""Run manifest and stage artifact lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from .io import atomic_write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunManifest:
    def __init__(self, path: Path, run_id: str, config: dict[str, Any], dataset_ids: tuple[str, ...]):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data["config_fingerprint"] != config["fingerprint"]:
                raise RuntimeError(
                    f"Run {run_id} was created with a different config; choose a new run ID"
                )
            if tuple(self.data.get("dataset_ids", config["datasets"])) != dataset_ids:
                raise RuntimeError(f"Run {run_id} was created with a different dataset selection")
        else:
            self.data = {
                "run_id": run_id,
                "protocol_version": config["protocol_version"],
                "config_path": str(config["config_path"]),
                "config_fingerprint": config["fingerprint"],
                "dataset_ids": list(dataset_ids),
                "created_at": utc_now(),
                "status": "partial",
                "stages": {},
            }
            self.save()

    def start(self, stage: str) -> None:
        self.data["stages"][stage] = {"status": "running", "started_at": utc_now()}
        self.save()

    def finish(self, stage: str, artifacts: list[Path], details: dict[str, Any]) -> None:
        record = self.data["stages"][stage]
        record.update(
            status="complete",
            finished_at=utc_now(),
            artifacts=[{"path": str(path), "size_bytes": path.stat().st_size} for path in artifacts],
            **details,
        )
        self.save()

    def fail(self, stage: str, message: str) -> None:
        self.data["stages"][stage].update(status="failed", finished_at=utc_now(), error=message)
        self.save()

    def complete(self, stage: str) -> bool:
        return self.data.get("stages", {}).get(stage, {}).get("status") == "complete"

    def set_status(self, status: str, **details: Any) -> None:
        self.data.update(status=status, **details)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.data)
