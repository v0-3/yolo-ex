"""Data models for export requests and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExportFormat(str, Enum):
    """Supported export formats."""

    COREML = "coreml"
    ENGINE = "engine"


@dataclass(slots=True)
class ExportRequest:
    """Normalized export request."""

    model_path: Path
    format: ExportFormat
    output_dir: Path | None = None
    imgsz: int = 640
    device: str | None = None
    half: bool = False
    int8: bool = False
    batch: int | None = None
    workspace: float | None = None
    nms: bool = False
    verbose: bool = False
    dry_run: bool = False


@dataclass(slots=True)
class ExportResult:
    """Export execution result."""

    format: ExportFormat
    input_model: Path
    output_path: Path | None
    backend: str
    details: dict[str, str] = field(default_factory=dict)
