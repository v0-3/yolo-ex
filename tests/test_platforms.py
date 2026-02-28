from __future__ import annotations

from types import SimpleNamespace

import pytest

import yolo_ex.platforms as platforms
from yolo_ex.errors import ExportValidationError
from yolo_ex.models import ExportFormat
from yolo_ex.platforms import PlatformTarget, detect_platform, preflight_for_format


def test_detect_platform_jetson(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "aarch64")
    assert detect_platform() is PlatformTarget.JETSON


def test_detect_platform_other(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "x86_64")
    assert detect_platform() is PlatformTarget.OTHER


def test_engine_preflight_rejects_non_jetson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "x86_64")

    with pytest.raises(ExportValidationError, match="supported only on Jetson"):
        preflight_for_format(ExportFormat.ENGINE)


def test_engine_preflight_missing_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "aarch64")

    def fake_import(name: str) -> SimpleNamespace:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(platforms.importlib, "import_module", fake_import)

    with pytest.raises(ExportValidationError, match="system-site-packages"):
        preflight_for_format(ExportFormat.ENGINE)


def test_engine_preflight_requires_tensorrt_not_tensorrt_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "aarch64")

    def fake_import(name: str) -> SimpleNamespace:
        if name == "tensorrt":
            raise ModuleNotFoundError(name)
        if name == "tensorrt_bindings":
            return SimpleNamespace()
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(platforms.importlib, "import_module", fake_import)

    with pytest.raises(ExportValidationError, match="tensorrt"):
        preflight_for_format(ExportFormat.ENGINE)


@pytest.mark.parametrize("missing_module", ["torch", "torchvision", "onnxruntime"])
def test_engine_preflight_requires_jetson_runtime_modules(
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "aarch64")

    def fake_import(name: str) -> SimpleNamespace:
        if name == missing_module:
            raise ModuleNotFoundError(name)
        return SimpleNamespace()

    monkeypatch.setattr(platforms.importlib, "import_module", fake_import)

    with pytest.raises(ExportValidationError, match=missing_module):
        preflight_for_format(ExportFormat.ENGINE)


def test_engine_preflight_requires_jetson_runtime_modules_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms.py_platform, "system", lambda: "Linux")
    monkeypatch.setattr(platforms.py_platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(platforms.importlib, "import_module", lambda name: SimpleNamespace())

    result = preflight_for_format(ExportFormat.ENGINE)
    assert result.target is PlatformTarget.JETSON
    assert result.warnings == []
