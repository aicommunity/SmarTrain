"""Matplotlib backend policy for headless Ultralytics training (no Tk/Qt GUI)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Final

_SAFE_BACKENDS_LOWER: Final[frozenset[str]] = frozenset(
    {"agg", "svg", "pdf", "ps", "template", "cairo"}
)


def configure_matplotlib_before_ultralytics() -> None:
    """Must run in each entry module before ``from ultralytics import YOLO``."""
    import matplotlib

    matplotlib.use("Agg")


def _looks_headless() -> bool:
    if (os.environ.get("DISPLAY") or "").strip():
        return False
    if (os.environ.get("WAYLAND_DISPLAY") or "").strip():
        return False
    return True


@dataclass(frozen=True)
class MatplotlibTrainingRuntime:
    headless_profile: bool
    matplotlib_backend: str
    force_ultralytics_plots_false: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_matplotlib_training_runtime(*, non_interactive: bool = False) -> MatplotlibTrainingRuntime:
    """Validate/fix matplotlib for train/test/resume; call at start of train_yolo / test_yolo / resume test."""
    import matplotlib

    headless = _looks_headless()
    profile = headless or non_interactive

    if profile:
        mpl_env = (os.environ.get("MPLBACKEND") or "").strip().lower()
        risky = not mpl_env or mpl_env in {
            "tkagg",
            "qt5agg",
            "qtagg",
            "gtk3agg",
            "gtk4agg",
            "wxagg",
        }
        if risky:
            os.environ["MPLBACKEND"] = "Agg"
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass

    backend = matplotlib.get_backend()
    backend_l = str(backend).lower()
    unsafe = bool(headless and backend_l not in _SAFE_BACKENDS_LOWER)
    force_plots = bool(non_interactive or unsafe)
    if unsafe:
        print(
            f"[WARN] Headless environment but matplotlib backend is {backend!r}; "
            "Ultralytics plots will be disabled (plots=False). "
            "Export MPLBACKEND=Agg before starting Python for a stable fix."
        )

    return MatplotlibTrainingRuntime(
        headless_profile=headless,
        matplotlib_backend=str(backend),
        force_ultralytics_plots_false=force_plots,
    )
