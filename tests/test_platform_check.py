from __future__ import annotations

import pytest

import yolo_ex.platform_check as platform_check
from yolo_ex.platform_check import (
    PackageCheckStatus,
    check_current_platform,
    render_platform_report,
)
from yolo_ex.platforms import PlatformTarget


def _patch_platform(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: PlatformTarget,
    system: str,
    machine: str,
) -> None:
    monkeypatch.setattr(platform_check, "detect_platform", lambda: target)
    monkeypatch.setattr(platform_check.py_platform, "system", lambda: system)
    monkeypatch.setattr(platform_check.py_platform, "machine", lambda: machine)


def test_check_current_platform_macos_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.MACOS,
        system="Darwin",
        machine="arm64",
    )

    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "coremltools": "9.0",
    }
    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.supported is True
    assert report.ok is True
    assert report.platform_target is PlatformTarget.MACOS
    assert all(check.status is PackageCheckStatus.OK for check in report.checks)


def test_check_current_platform_macos_missing_coremltools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.MACOS,
        system="Darwin",
        machine="arm64",
    )

    def fake_version(name: str) -> str:
        if name == "coremltools":
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return {
            "ultralytics": "8.4.14",
            "torch": "2.7.0",
            "torchvision": "0.22.0",
        }[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.ok is False
    coreml_check = next(check for check in report.checks if check.label == "coremltools")
    assert coreml_check.status is PackageCheckStatus.MISSING


def test_check_current_platform_linux_arm64_tensorrt_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )

    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.23.0",
        "tensorrt": "10.7.0",
    }

    def fake_version(name: str) -> str:
        if name not in versions:
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)

    def fake_import(name: str) -> object:
        if name == "tensorrt":
            raise ImportError("libnvinfer.so missing")
        return object()

    monkeypatch.setattr(platform_check.importlib, "import_module", fake_import)

    report = check_current_platform()

    assert report.supported is True
    assert report.ok is False
    tensorrt_import = next(
        check for check in report.checks if check.label == "TensorRT Python import"
    )
    assert tensorrt_import.status is PackageCheckStatus.MISSING
    assert "import failed" in tensorrt_import.message
    rendered = render_platform_report(report)
    assert "TensorRT Python import" not in rendered
    assert "[OK] TensorRT package" in rendered
    assert "libnvinfer.so missing" in rendered
    assert "warning: Jetson TensorRT checks assume" not in rendered
    assert "warning: If JetPack installs Python packages" not in rendered
    assert "Jetson tip:" in rendered
    assert "--system-site-packages" in rendered


def test_check_current_platform_linux_arm64_torch_version_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.8",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.23.0",
        "tensorrt": "10.7.0",
    }

    def fake_version(name: str) -> str:
        if name not in versions:
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    torch_check = next(check for check in report.checks if check.label == "torch")
    assert torch_check.status is PackageCheckStatus.OK


def test_check_current_platform_linux_arm64_does_not_accept_tensorrt_bindings_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.8",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.23.0",
        "tensorrt": "10.7.0",
    }

    def fake_version(name: str) -> str:
        if name not in versions:
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    def fake_import(name: str) -> object:
        if name == "tensorrt":
            raise ImportError("No module named 'tensorrt'")
        if name == "tensorrt_bindings":
            return object()
        return object()

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", fake_import)

    report = check_current_platform()

    assert report.ok is False
    tensorrt_import = next(
        check for check in report.checks if check.label == "TensorRT Python import"
    )
    assert tensorrt_import.status is PackageCheckStatus.MISSING
    assert tensorrt_import.import_name == "tensorrt"
    assert "import failed" in tensorrt_import.message


def test_check_current_platform_linux_arm64_tensorrt_metapackage_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )

    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.23.0",
        "tensorrt": "10.7.0",
    }

    def fake_version(name: str) -> str:
        if name not in versions:
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.ok is True
    tensorrt_package = next(check for check in report.checks if check.label == "TensorRT package")
    assert tensorrt_package.status is PackageCheckStatus.OK
    assert tensorrt_package.distribution == "tensorrt"
    rendered = render_platform_report(report)
    assert "[OK] TensorRT package expected=10.7.0 installed=10.7.0" in rendered
    assert "TensorRT Python import" not in rendered


def test_check_current_platform_linux_arm64_requires_tensorrt_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )

    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.23.0",
    }

    def fake_version(name: str) -> str:
        if name not in versions:
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.ok is False
    tensorrt_package = next(check for check in report.checks if check.label == "TensorRT package")
    assert tensorrt_package.status is PackageCheckStatus.MISSING
    assert tensorrt_package.distribution == "tensorrt"


def test_check_current_platform_linux_arm64_accepts_system_package_torch_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "onnxruntime-gpu": "1.99.0",
        "tensorrt": "10.7.0",
    }

    def fake_version(name: str) -> str:
        if name not in versions:
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.ok is True
    torch_check = next(check for check in report.checks if check.label == "torch")
    assert torch_check.status is PackageCheckStatus.OK
    assert torch_check.message == ""
    assert torch_check.expected_version == "2.5.0a0+872d972e41.nv24.08"
    assert torch_check.installed_version == "2.6.0"


def test_check_current_platform_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.OTHER,
        system="Linux",
        machine="x86_64",
    )

    report = check_current_platform()

    assert report.supported is False
    assert report.ok is False
    assert report.checks == []
    assert report.warnings


def test_check_current_platform_macos_torch_version_mismatch_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.MACOS,
        system="Darwin",
        machine="arm64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.9.0",
        "torchvision": "0.22.0",
        "coremltools": "9.0",
    }
    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    torch_check = next(check for check in report.checks if check.label == "torch")
    assert report.ok is True
    assert torch_check.status is PackageCheckStatus.OK
    assert torch_check.expected_version == "2.7.0"
    assert torch_check.installed_version == "2.9.0"
    assert torch_check.message == ""


def test_check_current_platform_macos_missing_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.MACOS,
        system="Darwin",
        machine="arm64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "coremltools": "9.0",
    }

    def fake_import(name: str) -> object:
        if name == "torch":
            raise ImportError("torch import failed")
        return object()

    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", fake_import)

    report = check_current_platform()

    assert report.ok is False
    torch_check = next(check for check in report.checks if check.label == "torch")
    assert torch_check.status is PackageCheckStatus.MISSING
    assert "import failed" in torch_check.message


def test_check_current_platform_linux_arm64_onnxruntime_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.23.0",
        "tensorrt": "10.7.0",
    }
    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    onnxruntime_check = next(check for check in report.checks if check.label == "onnxruntime")
    assert onnxruntime_check.status is PackageCheckStatus.OK
    assert onnxruntime_check.distribution == "onnxruntime-gpu"
    assert onnxruntime_check.import_name == "onnxruntime"
    assert onnxruntime_check.expected_version == "1.23.0"
    assert onnxruntime_check.installed_version == "1.23.0"
    rendered = render_platform_report(report)
    assert "[OK] onnxruntime (dist: onnxruntime-gpu) expected=1.23.0 installed=1.23.0" in rendered


def test_check_current_platform_linux_arm64_missing_onnxruntime_gpu_renders_distribution_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )

    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "tensorrt": "10.7.0",
    }

    def fake_version(name: str) -> str:
        if name not in versions:
            raise platform_check.importlib_metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(platform_check.importlib_metadata, "version", fake_version)
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.ok is False
    onnxruntime_check = next(check for check in report.checks if check.label == "onnxruntime")
    assert onnxruntime_check.status is PackageCheckStatus.MISSING
    assert onnxruntime_check.distribution == "onnxruntime-gpu"
    rendered = render_platform_report(report)
    assert (
        "[MISSING] onnxruntime (dist: onnxruntime-gpu) expected=1.23.0 - distribution not installed"
        in rendered
    )


def test_check_current_platform_linux_arm64_onnxruntime_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.23.0",
        "tensorrt": "10.7.0",
    }

    def fake_import(name: str) -> object:
        if name == "onnxruntime":
            raise ImportError("onnxruntime import failed")
        return object()

    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", fake_import)

    report = check_current_platform()

    assert report.ok is False
    onnxruntime_check = next(check for check in report.checks if check.label == "onnxruntime")
    assert onnxruntime_check.status is PackageCheckStatus.MISSING
    assert "import failed" in onnxruntime_check.message


def test_check_current_platform_linux_arm64_onnxruntime_version_mismatch_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_platform(
        monkeypatch,
        target=PlatformTarget.LINUX_ARM64,
        system="Linux",
        machine="aarch64",
    )
    versions = {
        "ultralytics": "8.4.14",
        "torch": "2.5.0a0+872d972e41.nv24.08",
        "torchvision": "0.20.0a0+afc54f7",
        "onnxruntime-gpu": "1.24.0",
        "tensorrt": "10.7.0",
    }
    monkeypatch.setattr(platform_check.importlib_metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(platform_check.importlib, "import_module", lambda name: object())

    report = check_current_platform()

    assert report.ok is True
    onnxruntime_check = next(check for check in report.checks if check.label == "onnxruntime")
    assert onnxruntime_check.status is PackageCheckStatus.OK
    assert onnxruntime_check.expected_version == "1.23.0"
    assert onnxruntime_check.installed_version == "1.24.0"
    assert onnxruntime_check.message == ""
