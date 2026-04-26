from __future__ import annotations

import sys

from smartrain import device_selector as ds


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
