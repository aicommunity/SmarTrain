"""Shared MFEL ConvModule shim for broken upstream forks (picklable top-level)."""

from __future__ import annotations


class MFELConvModuleShim:  # picklable top-level shim for broken forks
    def __init__(self, c1, c2, k=1, s=1, p=0, norm_cfg=None, act_cfg=None, **kwargs):
        import torch.nn as nn

        super().__init__()
        groups = int(kwargs.get("groups", 1) or 1)
        dilation = int(kwargs.get("dilation", 1) or 1)
        bias = bool(kwargs.get("bias", False))
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(int(c2))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def patch_mfel_missing_symbols(*, host_globals: dict | None = None) -> None:
    """Install ConvModule on ultralytics.nn.modules.block when missing.

    ``host_globals`` should be the launcher module ``globals()`` so the patched
    class remains picklable under that module name for DataLoader workers.
    """
    try:
        import torch.nn as nn
        from ultralytics.nn.modules import block as block_mod
    except Exception:
        return
    if getattr(block_mod, "ConvModule", None) is not None:
        return

    g = host_globals if host_globals is not None else globals()
    host_name = str(g.get("__name__", __name__))

    class _PatchedConvModule(MFELConvModuleShim, nn.Module):  # type: ignore[misc]
        pass

    _PatchedConvModule.__module__ = host_name
    _PatchedConvModule.__qualname__ = "MFELPatchedConvModule"
    g["MFELPatchedConvModule"] = _PatchedConvModule
    # Also expose shim on host for pickle of any residual references.
    g.setdefault("MFELConvModuleShim", MFELConvModuleShim)
    setattr(block_mod, "ConvModule", _PatchedConvModule)
