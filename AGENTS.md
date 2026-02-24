# Repository Guidelines

## Project Structure & Module Organization
- `src/yolo_ex/`: application code for the CLI, export flow, platform checks, models, and error handling.
- `tests/`: pytest test suite (CLI, exporter, platform checks).
- `pyproject.toml`: project metadata, dependencies, `ruff`, `mypy`, and `pytest` configuration.
- `README.md`: setup and usage examples.
- `ARCHITECTURE.md`, `STACK.md`: high-level project notes.

Keep new modules under `src/yolo_ex/` and place matching tests in `tests/` (for example, `src/yolo_ex/foo.py` -> `tests/test_foo.py`).

## Build, Test, and Development Commands
- `uv sync`: install runtime + dev dependencies for Python 3.10.
- `uv sync --index https://pypi.nvidia.com/simple`: required on Jetson (`linux/aarch64`) for TensorRT-related packages.
- `uv run pytest`: run all tests.
- `uv run ruff check .`: lint imports/style/errors.
- `uv run mypy src tests`: run strict type checks.
- `uv run yolo-ex export /path/model.pt --format coreml --dry-run`: validate export inputs without writing artifacts.
- `uv run yolo-ex-platform`: print platform compatibility diagnostics.

## Coding Style & Naming Conventions
- Python only; target `3.10`.
- Follow `ruff` defaults in this repo with max line length `100`.
- Use type hints consistently (`mypy` is strict).
- Names: modules/functions/variables `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Prefer small focused functions and reuse existing models/errors instead of adding ad hoc dicts/exceptions.

## Testing Guidelines
- Framework: `pytest` (`tests/` is the configured test path).
- Test files use `test_*.py`; test functions use `test_*` names.
- Cover CLI exit codes, stderr/stdout messages, and platform-specific validation paths.
- Use `monkeypatch`, `capsys`, and `tmp_path` for isolated CLI and filesystem behavior.

## Commit & Pull Request Guidelines
- Git history is minimal (`Initial commit`), so use short imperative commit subjects (for example, `Add CoreML workspace validation test`).
- Keep commits focused and logically grouped.
- PRs should include: purpose, platform tested (`macOS` or `Jetson/Linux aarch64`), commands run (`pytest`, `ruff`, `mypy`), and sample CLI output when behavior changes.

## Security & Configuration Tips
- Do not commit model files, secrets, or machine-specific absolute paths.
- Platform dependencies differ by OS/architecture; verify changes on the intended target before merging.
