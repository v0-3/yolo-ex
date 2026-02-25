# yolo-ex

Small Python CLI for exporting YOLO `.pt` models with a platform-aware workflow.

Supported outputs:

- TensorRT `.engine` (primarily Jetson Orin Nano)
- CoreML `.mlpackage` (primarily macOS)

## What This Tool Does

`yolo-ex` wraps Ultralytics export with:

- format-specific argument validation
- platform preflight checks (with actionable error messages)
- a `--dry-run` mode to validate the export plan before running
- a separate platform diagnostic command (`yolo-ex-platform`)

## Requirements

- Python `>=3.10,<3.11`
- `uv` for environment and dependency management

## Setup

### macOS (CoreML export)

```bash
uv sync --group mac
```

### Jetson Orin Nano (TensorRT export)

JetPack/system Python packages are used for TensorRT support/runtime compatibility. Project
Python dependencies should still be installed with the Jetson dependency group.

Create the venv with system site-packages so `uv run` can see JetPack-provided TensorRT modules,
then install the project dependencies with `uv`:

```bash
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --group jetson
```

Notes:

- TensorRT export still requires a compatible JetPack / CUDA / TensorRT runtime.
- The Jetson dependency group is used alongside JetPack/system TensorRT support.
- `coremltools` is installed only on macOS via the `mac` dependency group.

## Usage

### Export a model

```bash
uv run yolo-ex export /path/to/model.pt --format coreml --dry-run
uv run yolo-ex export /path/to/model.pt --format engine --workspace 4 --device 0
```

Common options:

- `--format {coreml,engine}` (required)
- `--output-dir /path/to/exports`
- `--imgsz 640`
- `--half`
- `--int8`
- `--nms`
- `--dry-run`
- `--verbose`

Engine-focused options:

- `--device` (for example `0` or `cpu`)
- `--batch`
- `--workspace` (GB, TensorRT only)

Validation notes:

- Input file must exist and use the `.pt` extension.
- `--workspace` is rejected unless `--format engine`.
- `--dry-run` still performs platform/dependency preflight checks.

### Platform check

Validate the current platform and dependency availability:

```bash
uv run yolo-ex-platform
```

On Jetson Orin Nano, complete the Jetson setup above first, then run the same command.

## Development

Install development tools:

```bash
uv sync --group dev
```

Run checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

## Project Docs

- `ARCHITECTURE.md`
- `STACK.md`
