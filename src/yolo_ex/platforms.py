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

    MACOS = "macos"
    LINUX_ARM64 = "linux_arm64"
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
    if system == "darwin":
        return PlatformTarget.MACOS
    if system == "linux" and machine in {"aarch64", "arm64"}:
        return PlatformTarget.LINUX_ARM64
    return PlatformTarget.OTHER


def preflight_for_format(export_format: ExportFormat) -> PreflightResult:
    """Validate platform/runtime requirements for the requested format."""
    target = detect_platform()
    warnings: list[str] = []

    if export_format is ExportFormat.COREML:
        if target is not PlatformTarget.MACOS:
            warnings.append(
                "CoreML export is intended for macOS. Continuing, but export may fail on "
                "this platform."
            )
        _require_module(
            "coremltools",
            "CoreML export requires `coremltools`. Install on macOS with `uv sync`.",
        )
    elif export_format is ExportFormat.ENGINE:
        if target is not PlatformTarget.LINUX_ARM64:
            warnings.append(
                "TensorRT export is intended for Jetson Orin Nano (Linux arm64). "
                "Continuing, but export may fail on this platform."
            )
        _require_module(
            "tensorrt",
            "TensorRT export requires `tensorrt-cu12-bindings==10.7.0.post1` from NVIDIA PyPI "
            "(https://pypi.nvidia.com) and a compatible JetPack/TensorRT runtime.",
        )

    return PreflightResult(target=target, warnings=warnings)


def _require_module(module_name: str, message: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ExportValidationError(message) from exc
