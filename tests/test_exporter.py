from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

import yolo_ex.exporter as exporter
from yolo_ex.errors import ExportExecutionError, ExportValidationError
from yolo_ex.exporter import build_export_kwargs, export_model, validate_request
from yolo_ex.models import ExportFormat, ExportRequest


def _pt_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.touch()
    return path


def test_validate_request_rejects_missing_file(tmp_path: Path) -> None:
    request = ExportRequest(model_path=tmp_path / "missing.pt", format=ExportFormat.COREML)
    with pytest.raises(ExportValidationError, match="not found"):
        validate_request(request)


def test_validate_request_rejects_non_pt(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.touch()
    request = ExportRequest(model_path=path, format=ExportFormat.COREML)
    with pytest.raises(ExportValidationError, match=".pt"):
        validate_request(request)


def test_validate_request_rejects_workspace_for_coreml(tmp_path: Path) -> None:
    request = ExportRequest(
        model_path=_pt_file(tmp_path),
        format=ExportFormat.COREML,
        workspace=4.0,
    )
    with pytest.raises(ExportValidationError, match="only valid"):
        validate_request(request)


def test_build_export_kwargs_for_engine(tmp_path: Path) -> None:
    request = ExportRequest(
        model_path=_pt_file(tmp_path),
        format=ExportFormat.ENGINE,
        output_dir=tmp_path / "exports",
        imgsz=320,
        device="0",
        half=True,
        int8=True,
        batch=2,
        workspace=8.0,
        nms=True,
    )

    kwargs = build_export_kwargs(request)

    assert kwargs["imgsz"] == 320
    assert kwargs["device"] == "0"
    assert kwargs["half"] is True
    assert kwargs["int8"] is True
    assert kwargs["batch"] == 2
    assert kwargs["workspace"] == 8.0
    assert kwargs["nms"] is True
    assert kwargs["project"] == str(tmp_path / "exports")
    assert kwargs["name"] == "model"


def test_build_export_kwargs_for_coreml_omits_engine_only_fields(tmp_path: Path) -> None:
    request = ExportRequest(
        model_path=_pt_file(tmp_path),
        format=ExportFormat.COREML,
        device="0",
        batch=2,
    )

    kwargs = build_export_kwargs(request)

    assert "device" not in kwargs
    assert "batch" not in kwargs
    assert "workspace" not in kwargs


def test_export_model_dry_run_skips_yolo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    request = ExportRequest(model_path=_pt_file(tmp_path), format=ExportFormat.COREML, dry_run=True)

    def fail_load() -> None:
        raise AssertionError("YOLO should not be loaded during dry run")

    monkeypatch.setattr(exporter, "_load_yolo_class", fail_load)

    result = export_model(request)
    assert result.output_path is None
    assert result.details["dry_run"] == "true"


def test_export_model_wraps_backend_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeModel:
        def export(self, **kwargs: object) -> str:
            raise RuntimeError("backend blew up")

    class FakeYolo:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def export(self, **kwargs: object) -> str:
            raise RuntimeError("backend blew up")

    monkeypatch.setattr(exporter, "_load_yolo_class", lambda: FakeYolo)

    request = ExportRequest(model_path=_pt_file(tmp_path), format=ExportFormat.ENGINE)
    with pytest.raises(ExportExecutionError, match="Ultralytics export failed"):
        export_model(request)


def test_export_model_returns_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "model.engine"

    class FakeYolo:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def export(self, **kwargs: object) -> str:
            return str(output)

    monkeypatch.setattr(exporter, "_load_yolo_class", lambda: FakeYolo)

    request = ExportRequest(model_path=_pt_file(tmp_path), format=ExportFormat.ENGINE)
    result = export_model(request)
    assert result.output_path == output


def test_export_model_engine_calls_tensorrt_compat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    class FakeYolo:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def export(self, **kwargs: object) -> str:
            return str(tmp_path / "model.engine")

    def fake_compat() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(exporter, "_ensure_tensorrt_module_compat", fake_compat)
    monkeypatch.setattr(exporter, "_load_yolo_class", lambda: FakeYolo)

    request = ExportRequest(model_path=_pt_file(tmp_path), format=ExportFormat.ENGINE)
    export_model(request)

    assert called is True


def test_export_model_coreml_skips_tensorrt_compat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeYolo:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def export(self, **kwargs: object) -> str:
            return str(tmp_path / "model.mlpackage")

    def fail_compat() -> None:
        raise AssertionError("TensorRT compat shim should not run for CoreML exports")

    monkeypatch.setattr(exporter, "_ensure_tensorrt_module_compat", fail_compat)
    monkeypatch.setattr(exporter, "_load_yolo_class", lambda: FakeYolo)

    request = ExportRequest(model_path=_pt_file(tmp_path), format=ExportFormat.COREML)
    result = export_model(request)

    assert result.output_path == tmp_path / "model.mlpackage"


def test_ensure_tensorrt_module_compat_aliases_tensorrt_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("tensorrt_bindings")

    def fake_import(name: str) -> object:
        if name == "tensorrt":
            raise ModuleNotFoundError(name)
        if name == "tensorrt_bindings":
            return fake_module
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.delitem(exporter.sys.modules, "tensorrt", raising=False)
    monkeypatch.setattr(exporter.importlib, "import_module", fake_import)

    exporter._ensure_tensorrt_module_compat()

    assert exporter.sys.modules["tensorrt"] is fake_module


def test_ensure_tensorrt_module_compat_noop_when_tensorrt_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("tensorrt")
    calls: list[str] = []

    def fake_import(name: str) -> object:
        calls.append(name)
        if name == "tensorrt":
            return fake_module
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(exporter.importlib, "import_module", fake_import)

    exporter._ensure_tensorrt_module_compat()

    assert calls == ["tensorrt"]
