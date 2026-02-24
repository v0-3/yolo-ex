"""Command-line interface for yolo-ex."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from yolo_ex.errors import ExportExecutionError, ExportValidationError
from yolo_ex.exporter import export_model
from yolo_ex.logging_utils import configure_logging
from yolo_ex.models import ExportFormat, ExportRequest
from yolo_ex.platforms import preflight_for_format

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="yolo-ex", description="YOLO model export tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export a YOLO .pt model")
    export_parser.add_argument("model_path", type=Path, help="Path to the source .pt model")
    export_parser.add_argument(
        "--format",
        "-f",
        required=True,
        choices=[fmt.value for fmt in ExportFormat],
        help="Target export format",
    )
    export_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Directory where exported artifacts are written",
    )
    export_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    export_parser.add_argument("--device", help="Device string for export (e.g. cpu, 0)")
    export_parser.add_argument("--half", action="store_true", help="Enable half precision")
    export_parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    export_parser.add_argument("--batch", type=int, help="Batch size")
    export_parser.add_argument(
        "--workspace",
        type=float,
        help="TensorRT workspace size in GB (engine export only)",
    )
    export_parser.add_argument("--nms", action="store_true", help="Include NMS in exported model")
    export_parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    export_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print export plan",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging(args.verbose)

    if args.command != "export":
        parser.error(f"Unsupported command: {args.command}")

    request = _request_from_args(args)
    try:
        preflight = preflight_for_format(request.format)
        for warning in preflight.warnings:
            print(f"warning: {warning}", file=sys.stderr)

        result = export_model(request)
    except ExportValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ExportExecutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if request.dry_run:
        print(
            f"Dry run OK: format={result.format.value} platform={preflight.target.value} "
            f"model={result.input_model}"
        )
        if "kwargs" in result.details:
            print(f"export kwargs: {result.details['kwargs']}")
        return 0

    print(
        f"Export complete: format={result.format.value} model={result.input_model} "
        f"output={result.output_path or 'unknown'}"
    )
    return 0


def _request_from_args(args: argparse.Namespace) -> ExportRequest:
    request = ExportRequest(
        model_path=args.model_path,
        format=ExportFormat(args.format),
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        batch=args.batch,
        workspace=args.workspace,
        nms=args.nms,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    LOGGER.debug("Resolved export request: %s", request)
    return request


if __name__ == "__main__":
    raise SystemExit(main())
