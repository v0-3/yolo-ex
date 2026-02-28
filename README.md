# yolo-ex

Small Python CLI for exporting YOLO `.pt` models with a platform-aware workflow.

Supported outputs:

- TensorRT `.engine` (primarily Jetson Orin Nano)

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

### Jetson Orin Nano (TensorRT export)

JetPack/system Python packages are used for TensorRT support/runtime compatibility. Project
Python dependencies are installed with Jetson as a default `uv` dependency group.

Create the venv with system site-packages so `uv run` can see JetPack-provided TensorRT modules,
then install the project dependencies with `uv`:

```bash
uv venv --python /usr/bin/python3 --system-site-packages
uv sync
```

Notes:

- TensorRT export still requires a compatible JetPack / CUDA / TensorRT runtime.
- The default dependency groups are `dev` and `jetson`.
- Jetson dependencies are used alongside JetPack/system TensorRT support.

## Usage

### Export a model

```bash
uv run yolo-ex export /path/to/model.pt --format engine --dry-run
uv run yolo-ex export /path/to/model.pt --format engine --workspace 4 --device 0
```

Common options:

- `--format engine` (required)
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
- `--dry-run` still performs platform/dependency preflight checks.

### Platform check

Validate the current platform and dependency availability:

```bash
uv run yolo-ex-platform
```

On Jetson Orin Nano, complete the Jetson setup above first, then run the same command.

## Troubleshooting

### CPU torch selected unexpectedly

If `uv run` reports a CPU-only Torch build (for example `2.7.0+cpu`) or `torch.cuda.is_available()`
is `False`, resync and verify:

```bash
uv sync --group jetson
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

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
