from __future__ import annotations

from pathlib import Path

import pytest

import yolo_ex.cli as cli
from yolo_ex.errors import ExportExecutionError
from yolo_ex.models import ExportFormat, ExportResult
from yolo_ex.platforms import PlatformTarget, PreflightResult


def test_cli_invalid_format_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["export", "model.pt", "--format", "onnx"])
    assert excinfo.value.code == 2


def test_cli_coreml_format_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["export", "model.pt", "--format", "coreml"])
    assert excinfo.value.code == 2


def test_cli_dry_run_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "preflight_for_format",
        lambda fmt: PreflightResult(target=PlatformTarget.JETSON, warnings=["heads up"]),
    )

    def fake_export_model(request: object) -> ExportResult:
        return ExportResult(
            format=ExportFormat.ENGINE,
            input_model=Path("model.pt"),
            output_path=None,
            backend="ultralytics",
            details={"dry_run": "true", "kwargs": '{"imgsz": 640}'},
        )

    monkeypatch.setattr(cli, "export_model", fake_export_model)

    exit_code = cli.main(["export", "model.pt", "--format", "engine", "--dry-run"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "warning:" in captured.err.lower()
    assert "Dry run OK" in captured.out
    assert "platform=jetson" in captured.out


def test_cli_exporter_error_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "preflight_for_format",
        lambda fmt: PreflightResult(target=PlatformTarget.JETSON),
    )
    monkeypatch.setattr(
        cli,
        "export_model",
        lambda request: (_ for _ in ()).throw(ExportExecutionError("boom")),
    )

    exit_code = cli.main(["export", "model.pt", "--format", "engine", "--dry-run"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "boom" in captured.err
