from __future__ import annotations

from types import SimpleNamespace

import pytest

import yolo_ex.platforms as platforms
from yolo_ex.errors import ExportValidationError
from yolo_ex.models import ExportFormat
from yolo_ex.platforms import PlatformTarget, detect_platform, preflight_for_format


def test_detect_platform_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "arm64")
    assert detect_platform() is PlatformTarget.MACOS


def test_detect_platform_linux_arm64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "aarch64")
    assert detect_platform() is PlatformTarget.LINUX_ARM64


def test_detect_platform_other(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "x86_64")
    assert detect_platform() is PlatformTarget.OTHER


def test_coreml_preflight_missing_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "arm64")

    def fake_import(name: str) -> SimpleNamespace:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(platforms.importlib, "import_module", fake_import)

    with pytest.raises(ExportValidationError, match="coremltools"):
        preflight_for_format(ExportFormat.COREML)


def test_engine_preflight_missing_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "aarch64")

    def fake_import(name: str) -> SimpleNamespace:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(platforms.importlib, "import_module", fake_import)

    with pytest.raises(ExportValidationError, match="tensorrt-cu12-bindings==10.7.0.post1"):
        preflight_for_format(ExportFormat.ENGINE)


def test_engine_preflight_warns_on_non_jetson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platforms.importlib, "import_module", lambda name: SimpleNamespace())

    result = preflight_for_format(ExportFormat.ENGINE)
    assert result.target is PlatformTarget.MACOS
    assert result.warnings
