from __future__ import annotations

import sys

from smartrain.core.runtime import device_selector as ds


def test_default_device_value_prefers_gpu0(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(idx: int) -> str:
            return f"GPU-{idx}"

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert ds.default_device_value() == "0"


def test_default_device_value_falls_back_to_cpu(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert ds.default_device_value() == "cpu"


def test_resolve_device_request_by_cuda_tokens(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(idx: int) -> str:
            return f"RTX-{idx}"

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert ds.resolve_device_request("cuda:1") == "1"
    assert ds.resolve_device_request("gpu1") == "1"
    assert ds.resolve_device_request("cpu") == "cpu"


def test_resolve_device_request_by_gpu_name(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(idx: int) -> str:
            return "NVIDIA GeForce RTX 3060" if idx == 0 else "NVIDIA GeForce RTX 4090"

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    assert ds.resolve_device_request("rtx 4090") == "1"


def test_validate_device_available_checks_index(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

    class _FakeTorch:
        __version__ = "fake"
        cuda = _FakeCuda()
        version = type("_V", (), {"cuda": "12.1"})()

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    try:
        ds.validate_device_available("1")
    except RuntimeError as exc:
        assert "available GPU indices are 0..0" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for unavailable GPU index")
