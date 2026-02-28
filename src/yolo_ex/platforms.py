"""Platform detection and format-specific preflight checks."""

from __future__ import annotations

import importlib
import platform as py_platform
from dataclasses import dataclass, field
from enum import Enum

from yolo_ex.errors import ExportValidationError
from yolo_ex.models import ExportFormat


class PlatformTarget(str, Enum):
    """Supported runtime platform buckets."""

    JETSON = "jetson"
    OTHER = "other"


@dataclass(slots=True)
class PreflightResult:
    """Preflight result including warnings."""

    target: PlatformTarget
    warnings: list[str] = field(default_factory=list)


def detect_platform() -> PlatformTarget:
    """Detect the current platform bucket."""
    system = py_platform.system().lower()
    machine = py_platform.machine().lower()
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return PlatformTarget.JETSON
    return PlatformTarget.OTHER


def preflight_for_format(export_format: ExportFormat) -> PreflightResult:
    """Validate platform/runtime requirements for the requested format."""
    target = detect_platform()
    if export_format is not ExportFormat.ENGINE:
        raise ExportValidationError("Only --format engine is supported in this Jetson-only build.")
    if target is not PlatformTarget.JETSON:
        raise ExportValidationError(
            "TensorRT export is supported only on Jetson (Linux arm64). "
            "Run this command on a Jetson device with JetPack/TensorRT installed."
        )

    _require_module(
        "tensorrt",
        "TensorRT export requires the Jetson TensorRT Python package (`tensorrt`) and a "
        "compatible JetPack/TensorRT runtime. If JetPack provides system Python packages, "
        "create the project venv with `uv venv --python /usr/bin/python3 "
        "--system-site-packages` before `uv sync`.",
    )
    _require_module(
        "torch",
        "TensorRT export on Jetson requires `torch`. If JetPack provides system Python "
        "packages, create the project venv with `uv venv --python /usr/bin/python3 "
        "--system-site-packages` before `uv sync`.",
    )
    _require_module(
        "torchvision",
        "TensorRT export on Jetson requires `torchvision`. If JetPack provides system "
        "Python packages, create the project venv with `uv venv --python /usr/bin/python3 "
        "--system-site-packages` before `uv sync`.",
    )
    _require_module(
        "onnxruntime",
        "TensorRT export on Jetson requires `onnxruntime`. If JetPack provides system "
        "Python packages, create the project venv with `uv venv --python /usr/bin/python3 "
        "--system-site-packages` before `uv sync`.",
    )

    return PreflightResult(target=target, warnings=[])


def _require_module(module_name: str, message: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ExportValidationError(message) from exc
