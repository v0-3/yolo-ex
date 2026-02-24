from __future__ import annotations

import pytest

import yolo_ex.platform_check as platform_check
from yolo_ex.platform_check import (
    PackageCheckStatus,
    check_current_platform,
    render_platform_report,
)
from yolo_ex.platforms import PlatformTarget


def _patch_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: PlatformTarget,
    system: str,
    machine: str,
) -> None:
    monkeypatch.setattr(platform_check, "detect_platform", lambda: target)
    monkeypatch.setattr(platform_check.py_platform, "system", lambda: system)
    monkeypatch.setattr(platform_check.py_platform, "machine", lambda: machine)


def test_check_current_platform_macos_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.MACOS,
        system="Darwin",
        machine="arm64",
    )

    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "coremltools": "9.0",
    }
    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.supported is True
    assert report.ok is True
    assert report.platform_target is PlatformTarget.MACOS
    assert all(check.status is PackageCheckStatus.OK for check in report.checks)


def test_check_current_platform_macos_missing_coremltools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.MACOS,
        system="Darwin",
        machine="arm64",
    )

    def fake_version(name: str) -> str:
        if name == "coremltools":
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return {
            "ultralytics": "8.4.14",
            "torch": "2.7.0",
            "torchvision": "0.22.0",
        }[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.ok is False
    coreml_check = next(check for check in report.checks if check.label == "coremltools")
    assert coreml_check.status is PackageCheckStatus.MISSING


def test_check_current_platform_linux_arm64_tensorrt_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )

    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "tensorrt-cu12-bindings": "10.7.0.post1",
    }
    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])

    def fake_import(name: str) -> object:
        if name == "tensorrt":
            raise ImportError("libnvinfer.so missing")
        return object()

    monkeypatch.setattr(platform_check.importlib, "import_module", fake_import)

    report = check_current_platform()

    assert report.supported is True
    assert report.ok is False
    tensorrt_import = next(
        check for check in report.checks if check.label == "TensorRT Python import"
    )
    assert tensorrt_import.status is PackageCheckStatus.MISSING
    assert "import failed" in tensorrt_import.message
    rendered = render_platform_report(report)
    assert "Jetson tip:" in rendered


def test_check_current_platform_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.OTHER,
        system="Linux",
        machine="x86_64",
    )

    report = check_current_platform()

    assert report.supported is False
    assert report.ok is False
    assert report.checks == []
    assert report.warnings


def test_check_current_platform_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.MACOS,
        system="Darwin",
        machine="arm64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.9.0",
        "torchvision": "0.22.0",
        "coremltools": "9.0",
    }
    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    torch_check = next(check for check in report.checks if check.label == "torch")
    assert torch_check.status is PackageCheckStatus.MISMATCH
