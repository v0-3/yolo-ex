"""Export request validation and Ultralytics export backend."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from yolo_ex.errors import ExportExecutionError, ExportValidationError
from yolo_ex.models import ExportFormat, ExportRequest, ExportResult


def export_model(request: ExportRequest) -> ExportResult:
    """Export a YOLO model using the Ultralytics backend."""
    validate_request(request)
    export_kwargs = build_export_kwargs(request)

    if request.dry_run:
        return ExportResult(
            format=request.format,
            input_model=request.model_path,
            output_path=None,
            backend="ultralytics",
            details={"dry_run": "true", "kwargs": json.dumps(export_kwargs, sort_keys=True)},
        )

    if request.format is ExportFormat.ENGINE:
        _ensure_tensorrt_module_compat()

    yolo_cls = _load_yolo_class()

    try:
        model = yolo_cls(str(request.model_path))
        exported = model.export(format=request.format.value, **export_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise ExportExecutionError(f"Ultralytics export failed: {exc}") from exc

    output_path = _normalize_output_path(exported)
    details = {"dry_run": "false", "kwargs": json.dumps(export_kwargs, sort_keys=True)}
    return ExportResult(
        format=request.format,
        input_model=request.model_path,
        output_path=output_path,
        backend="ultralytics",
        details=details,
    )


def validate_request(request: ExportRequest) -> None:
    """Validate an export request."""
    if not request.model_path.exists():
        raise ExportValidationError(f"Model file not found: {request.model_path}")
    if request.model_path.suffix.lower() != ".pt":
        raise ExportValidationError("Input model must use the .pt extension.")
    if request.imgsz <= 0:
        raise ExportValidationError("--imgsz must be greater than zero.")
    if request.batch is not None and request.batch <= 0:
        raise ExportValidationError("--batch must be greater than zero.")
    if request.workspace is not None and request.workspace <= 0:
        raise ExportValidationError("--workspace must be greater than zero.")
    if request.format is ExportFormat.COREML and request.workspace is not None:
        raise ExportValidationError("--workspace is only valid with --format engine.")


def build_export_kwargs(request: ExportRequest) -> dict[str, Any]:
    """Build Ultralytics export kwargs for the selected format."""
    kwargs: dict[str, Any] = {
        "imgsz": request.imgsz,
        "half": request.half,
        "int8": request.int8,
        "nms": request.nms,
    }

    if request.output_dir is not None:
        kwargs["project"] = str(request.output_dir)
        kwargs["name"] = request.model_path.stem

    if request.format is ExportFormat.ENGINE:
        if request.device is not None:
            kwargs["device"] = request.device
        if request.batch is not None:
            kwargs["batch"] = request.batch
        if request.workspace is not None:
            kwargs["workspace"] = request.workspace
    elif request.format is ExportFormat.COREML:
        # CoreML export in this tool intentionally omits device/batch/workspace.
        pass

    return kwargs


def _load_yolo_class() -> Any:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    return YOLO


def _ensure_tensorrt_module_compat() -> None:
    """Alias ``tensorrt_bindings`` to ``tensorrt`` when Jetson bindings omit the canonical module."""
    try:
        importlib.import_module("tensorrt")
        return
    except ImportError:
        pass

    try:
        tensorrt_bindings = importlib.import_module("tensorrt_bindings")
    except ImportError:
        return

    sys.modules.setdefault("tensorrt", tensorrt_bindings)


def _normalize_output_path(exported: object) -> Path | None:
    if isinstance(exported, Path):
        return exported
    if isinstance(exported, str):
        return Path(exported)
    return None
