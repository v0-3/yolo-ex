# yolo-ex
yolo model export tool

Small Python CLI to export YOLO `.pt` models to:

- TensorRT `.engine` (Jetson Orin Nano / Linux `aarch64`)
- CoreML `.mlpackage` (macOS)

## Setup

The project uses `uv` and targets Python `3.10.x`.

```bash
uv sync
```

Notes:

- On macOS, `uv sync` should work without adding the NVIDIA index.
- On Jetson Orin Nano (Linux `aarch64`), add NVIDIA PyPI when syncing:

```bash
uv sync --index https://pypi.nvidia.com/simple
```

- `tensorrt-cu12-bindings==10.7.0.post1` is pinned for Jetson (`linux/aarch64`).
- TensorRT export still depends on a compatible JetPack / CUDA / TensorRT runtime on Jetson.
- `coremltools` is installed only on macOS.

## Usage

```bash
uv run yolo-ex export /path/to/model.pt --format coreml --dry-run
uv run yolo-ex export /path/to/model.pt --format engine --workspace 4 --device 0
```

## Platform Check

Validate the current device platform and installed package versions:

```bash
uv run yolo-ex-platform
```

On Jetson Orin Nano, install with the NVIDIA index first:

```bash
uv sync --index https://pypi.nvidia.com/simple
uv run yolo-ex-platform
```
