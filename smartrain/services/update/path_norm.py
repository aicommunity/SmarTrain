"""Portable path helpers for ``smartrain update`` (relative POSIX only)."""

from __future__ import annotations

from pathlib import Path

from smartrain.core.runtime.workspace_paths import WorkspaceLayout

_WORKSPACE_MARKERS = ("models/", "runs/", "datasets/")


def is_abs_like(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    if v.startswith("/"):
        return True
    if len(v) >= 3 and v[1] == ":" and v[2] in "/\\":
        return True
    return False


def needs_path_rewrite(value: str) -> bool:
    """True when a stored path is absolute-like or uses Windows separators."""
    v = (value or "").strip()
    if not v:
        return False
    if is_abs_like(v):
        return True
    if "\\" in v:
        return True
    return False


def to_posix_rel(value: str) -> str:
    return (value or "").strip().replace("\\", "/")


def workspace_rel_posix(layout: WorkspaceLayout, path: Path) -> str:
    root = Path(layout.root).resolve()
    p = path.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _remap_legacy_artifact_tail(suffix: str, *, release_dir: Path | None, root: Path) -> str:
    """Map legacy ``.../train`` / ``.../test`` tails to canonical dirs when present."""
    s = suffix.replace("\\", "/").rstrip("/")
    if s.endswith("/train") or s == "train":
        base = s[: -len("train")].rstrip("/") if s.endswith("train") else ""
        for cand_tail in ("train-ultralytics", "train"):
            rel = f"{base}/{cand_tail}" if base else cand_tail
            if (root / rel).is_dir() or (
                release_dir is not None and (release_dir / cand_tail).is_dir()
            ):
                if release_dir is not None and (release_dir / cand_tail).is_dir():
                    try:
                        return (release_dir / cand_tail).resolve().relative_to(root).as_posix()
                    except ValueError:
                        pass
                if (root / rel).is_dir():
                    return rel
    if s.endswith("/test") or s == "test":
        base = s[: -len("test")].rstrip("/") if s.endswith("test") else ""
        for cand_tail in ("tests", "test"):
            rel = f"{base}/{cand_tail}" if base else cand_tail
            if (root / rel).is_dir() or (
                release_dir is not None and (release_dir / cand_tail).is_dir()
            ):
                if release_dir is not None and (release_dir / cand_tail).is_dir():
                    try:
                        return (release_dir / cand_tail).resolve().relative_to(root).as_posix()
                    except ValueError:
                        pass
                if (root / rel).is_dir():
                    return rel
    return s


def _suffix_candidates(normalized: str) -> list[str]:
    out: list[str] = []
    for marker in _WORKSPACE_MARKERS:
        start = 0
        while True:
            idx = normalized.find(marker, start)
            if idx < 0:
                break
            out.append(normalized[idx:])
            start = idx + 1
    return out


def normalize_stored_path(
    layout: WorkspaceLayout,
    value: str,
    *,
    release_dir: Path | None = None,
    must_exist: bool = False,
) -> str | None:
    """
    Return a workspace-relative POSIX path, or ``None`` if remap is impossible.

    Handles drive-letter abs, POSIX abs from other machines, and backslash relatives.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    root = Path(layout.root).resolve()
    normalized = to_posix_rel(raw)

    if not is_abs_like(normalized):
        remapped = _remap_legacy_artifact_tail(normalized, release_dir=release_dir, root=root)
        cand = root / remapped
        if must_exist and not cand.exists():
            return None
        return remapped

    # Native absolute under this workspace
    if not normalized.startswith("/"):
        try:
            rp = Path(raw).expanduser().resolve()
            if rp.is_relative_to(root):
                rel = rp.relative_to(root).as_posix()
                return _remap_legacy_artifact_tail(rel, release_dir=release_dir, root=root)
        except Exception:
            pass

    for suffix in _suffix_candidates(normalized):
        remapped = _remap_legacy_artifact_tail(suffix, release_dir=release_dir, root=root)
        cand = root / remapped
        if cand.exists() or not must_exist:
            if must_exist and not cand.exists():
                continue
            if cand.exists() or release_dir is not None:
                return remapped
            # Allow non-existing relative when suffix is clearly under workspace markers
            return remapped

    if release_dir is not None:
        for tail, canon in (("train", "train-ultralytics"), ("test", "tests"), ("train-ultralytics", "train-ultralytics"), ("tests", "tests")):
            if normalized.rstrip("/").endswith("/" + tail) or normalized.rstrip("/") == tail:
                target = release_dir / canon
                if target.is_dir() or (release_dir / tail).is_dir():
                    use = target if target.is_dir() else release_dir / tail
                    try:
                        return use.resolve().relative_to(root).as_posix()
                    except ValueError:
                        pass

    return None


def resolve_stored_path(layout: WorkspaceLayout | None, value: str, *, anchor: Path | None = None) -> Path | None:
    """Resolve a stored abs/relative path to an absolute Path when possible."""
    raw = (value or "").strip()
    if not raw:
        return None
    if layout is not None:
        rel = normalize_stored_path(layout, raw, must_exist=False)
        if rel:
            return (Path(layout.root) / rel).resolve()
    if is_abs_like(raw) and not raw.startswith("/"):
        try:
            return Path(raw).expanduser().resolve()
        except Exception:
            return None
    if anchor is not None:
        ws = None
        for anc in [anchor, *anchor.resolve().parents]:
            if (anc / "models").is_dir() and ((anc / "runs").is_dir() or (anc / "datasets").is_dir()):
                ws = anc
                break
        if ws is not None:
            for suffix in _suffix_candidates(to_posix_rel(raw)):
                cand = ws / suffix
                if cand.exists():
                    return cand.resolve()
            if not is_abs_like(raw):
                cand = ws / to_posix_rel(raw)
                return cand.resolve()
    return None
