"""Shared execution context passed to every registered stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageContext:
    config: dict[str, Any]
    run_id: str
    run_root: Path
    stage: str
    dataset_ids: tuple[str, ...]

    @property
    def stage_dir(self) -> Path:
        return self.run_root / "stages" / self.stage

    def upstream(self, stage: str, *parts: str) -> Path:
        return self.run_root.joinpath("stages", stage, *parts)
