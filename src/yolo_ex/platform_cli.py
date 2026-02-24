"""CLI entrypoint for platform setup validation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from yolo_ex.logging_utils import configure_logging
from yolo_ex.platform_check import check_current_platform, render_platform_report


def build_parser() -> argparse.ArgumentParser:
    """Build the platform check CLI parser."""
    parser = argparse.ArgumentParser(
        prog="yolo-ex-platform",
        description="Detect platform and validate required package versions",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run platform detection and package version checks."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(args.verbose)

    report = check_current_platform()
    print(render_platform_report(report))

    if not report.supported:
        return 2
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
