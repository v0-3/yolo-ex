"""Platform detection and package version checks for local setup validation."""

from __future__ import annotations

import importlib
import logging
import platform as py_platform
import warnings
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata as importlib_metadata

from yolo_ex.platforms import PlatformTarget, detect_platform

ULTRALYTICS_VERSION = "8.4.14"
MACOS_TORCH_VERSION = "2.7.0"
MACOS_TORCHVISION_VERSION = "0.22.0"
JETSON_TORCH_VERSION = "2.5.0a0+872d972e41.nv24.08"
JETSON_TORCHVISION_VERSION = "0.20.0a0+afc54f7"
JETSON_TENSORRT_BINDINGS_VERSION = "10.7.0.post1"


class PackageCheckStatus(str, Enum):
    """Status for a package version/import check."""

    OK = "ok"
    MISSING = "missing"
    MISMATCH = "mismatch"
    SKIPPED = "skipped"


@dataclass(slots=True)
class PackageCheck:
    """Result of checking one package requirement."""

    label: str
    distribution: str | None
    import_name: str | None
    expected_version: str | None
    installed_version: str | None
    status: PackageCheckStatus
    message: str


@dataclass(slots=True)
class PlatformCheckReport:
    """Complete setup validation result for the current platform."""

    platform_target: PlatformTarget
    platform_details: str
    supported: bool
    checks: list[PackageCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: bool = False


def check_current_platform() -> PlatformCheckReport:
    """Detect the current platform and validate required package versions."""
    target = detect_platform()
    details = f"{py_platform.system()} {py_platform.machine()}"
    warnings: list[str] = []

    if target is PlatformTarget.OTHER:
        warnings.append(
            "Unsupported platform for yolo-ex exports. Supported targets are macOS and Linux arm64 "
            "(Jetson Orin Nano)."
        )
        return PlatformCheckReport(
            platform_target=target,
            platform_details=details,
            supported=False,
            warnings=warnings,
            ok=False,
        )

    checks: list[PackageCheck] = [
        _check_exact_version("ultralytics", "ultralytics", ULTRALYTICS_VERSION)
    ]

    if target is PlatformTarget.MACOS:
        checks.extend(
            [
                _check_exact_version("torch", "torch", MACOS_TORCH_VERSION),
                _check_exact_version("torchvision", "torchvision", MACOS_TORCHVISION_VERSION),
                _check_presence_and_import("coremltools", "coremltools", "coremltools"),
            ]
        )
    elif target is PlatformTarget.LINUX_ARM64:
        checks.extend(
            [
                _check_exact_version("torch", "torch", JETSON_TORCH_VERSION),
                _check_exact_version("torchvision", "torchvision", JETSON_TORCHVISION_VERSION),
                _check_exact_version(
                    "TensorRT bindings",
                    "tensorrt-cu12-bindings",
                    JETSON_TENSORRT_BINDINGS_VERSION,
                ),
                _check_import_only("TensorRT Python import", "tensorrt"),
            ]
        )
        warnings.append(
            "Jetson TensorRT checks assume JetPack-compatible CUDA/TensorRT runtime is installed."
        )

    ok = all(check.status is PackageCheckStatus.OK for check in checks)
    return PlatformCheckReport(
        platform_target=target,
        platform_details=details,
        supported=True,
        checks=checks,
        warnings=warnings,
        ok=ok,
    )


def render_platform_report(report: PlatformCheckReport) -> str:
    """Render a human-readable platform check report."""
    lines: list[str] = [
        f"Platform check: {report.platform_target.value} ({report.platform_details})",
        f"Status: {_status_label(report)}",
    ]

    for check in report.checks:
        expected = (
            f" expected={check.expected_version}" if check.expected_version is not None else ""
        )
        installed = (
            f" installed={check.installed_version}" if check.installed_version is not None else ""
        )
        suffix = f" - {check.message}" if check.message else ""
        lines.append(f"[{check.status.name}] {check.label}{expected}{installed}{suffix}")

    for warning in report.warnings:
        lines.append(f"warning: {warning}")

    if report.platform_target is PlatformTarget.LINUX_ARM64 and not report.ok:
        lines.append("Jetson tip: run `uv sync --index https://pypi.nvidia.com/simple`.")

    return "\n".join(lines)


def _status_label(report: PlatformCheckReport) -> str:
    if not report.supported:
        return "UNSUPPORTED"
    if report.ok:
        return "OK"
    return "FAILED"


def _check_exact_version(label: str, distribution: str, expected_version: str) -> PackageCheck:
    installed_version = _get_distribution_version(distribution)
    if installed_version is None:
        return PackageCheck(
            label=label,
            distribution=distribution,
            import_name=None,
            expected_version=expected_version,
            installed_version=None,
            status=PackageCheckStatus.MISSING,
            message="distribution not installed",
        )
    if installed_version != expected_version:
        return PackageCheck(
            label=label,
            distribution=distribution,
            import_name=None,
            expected_version=expected_version,
            installed_version=installed_version,
            status=PackageCheckStatus.MISMATCH,
            message="version mismatch",
        )
    return PackageCheck(
        label=label,
        distribution=distribution,
        import_name=None,
        expected_version=expected_version,
        installed_version=installed_version,
        status=PackageCheckStatus.OK,
        message="",
    )


def _check_presence_and_import(label: str, distribution: str, import_name: str) -> PackageCheck:
    installed_version = _get_distribution_version(distribution)
    if installed_version is None:
        return PackageCheck(
            label=label,
            distribution=distribution,
            import_name=import_name,
            expected_version=None,
            installed_version=None,
            status=PackageCheckStatus.MISSING,
            message="distribution not installed",
        )
    try:
        _import_checked_module(import_name)
    except ImportError as exc:
        return PackageCheck(
            label=label,
            distribution=distribution,
            import_name=import_name,
            expected_version=None,
            installed_version=installed_version,
            status=PackageCheckStatus.MISSING,
            message=f"installed but import failed: {exc}",
        )
    return PackageCheck(
        label=label,
        distribution=distribution,
        import_name=import_name,
        expected_version=None,
        installed_version=installed_version,
        status=PackageCheckStatus.OK,
        message="",
    )


def _import_checked_module(import_name: str) -> None:
    if import_name != "coremltools":
        importlib.import_module(import_name)
        return

    # coremltools emits a compatibility warning for unsupported torch versions on import.
    # The platform checker reports version mismatches explicitly, so suppress that noise here.
    logger = logging.getLogger("coremltools")
    previous_disabled = logger.disabled
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Torch version .* has not been tested with coremltools.*",
        )
        try:
            logger.disabled = True
            importlib.import_module(import_name)
        finally:
            logger.disabled = previous_disabled


def _check_import_only(label: str, import_name: str) -> PackageCheck:
    try:
        importlib.import_module(import_name)
    except ImportError as exc:
        return PackageCheck(
            label=label,
            distribution=None,
            import_name=import_name,
            expected_version=None,
            installed_version=None,
            status=PackageCheckStatus.MISSING,
            message=f"import failed: {exc}",
        )
    return PackageCheck(
        label=label,
        distribution=None,
        import_name=import_name,
        expected_version=None,
        installed_version=None,
        status=PackageCheckStatus.OK,
        message="",
    )


def _get_distribution_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None
