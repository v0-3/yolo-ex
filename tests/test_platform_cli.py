from __future__ import annotations

import yolo_ex.platform_cli as platform_cli
from yolo_ex.platform_check import PlatformCheckReport
from yolo_ex.platforms import PlatformTarget


def _report(*, target: PlatformTarget, supported: bool, ok: bool) -> PlatformCheckReport:
    return PlatformCheckReport(
        platform_target=target,
        platform_details="TestOS testarch",
        supported=supported,
        checks=[],
        warnings=[],
        ok=ok,
    )


def test_platform_cli_returns_zero_on_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        platform_cli,
        "check_current_platform",
        lambda: _report(target=PlatformTarget.MACOS, supported=True, ok=True),
    )
    monkeypatch.setattr(platform_cli, "render_platform_report", lambda report: "ok report")

    exit_code = platform_cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok report" in captured.out


def test_platform_cli_returns_one_on_failed_checks(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        platform_cli,
        "check_current_platform",
        lambda: _report(target=PlatformTarget.MACOS, supported=True, ok=False),
    )
    monkeypatch.setattr(platform_cli, "render_platform_report", lambda report: "bad report")

    exit_code = platform_cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "bad report" in captured.out


def test_platform_cli_returns_two_on_unsupported_platform(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        platform_cli,
        "check_current_platform",
        lambda: _report(target=PlatformTarget.OTHER, supported=False, ok=False),
    )
    monkeypatch.setattr(platform_cli, "render_platform_report", lambda report: "unsupported")

    exit_code = platform_cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unsupported" in captured.out
