from __future__ import annotations

import types

from smartrain import tensorrt_checks as tc


def test_check_python_cuda_runtime_ready_ok(monkeypatch):
    fake_cuda_pkg = types.SimpleNamespace(cudart=object())
    monkeypatch.setitem(__import__("sys").modules, "cuda", fake_cuda_pkg)
    ok, reason = tc.check_python_cuda_runtime_ready()
    assert ok is True
    assert reason == ""


def test_check_python_cuda_runtime_ready_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cuda":
            raise ImportError("cannot import name 'cudart' from 'cuda'")
        if name == "cuda.bindings":
            raise ImportError("cannot import name 'runtime' from 'cuda.bindings'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    ok, reason = tc.check_python_cuda_runtime_ready()
    assert ok is False
    assert "python CUDA runtime is unavailable" in reason
    assert "cudart" in reason
