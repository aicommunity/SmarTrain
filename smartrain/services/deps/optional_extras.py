from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import smartrain
from smartrain.services.reporting.document_export import check_pandoc_ready, check_weasyprint_ready

PACKAGE_NAME = "smartrain"

KNOWN_OPTIONAL_EXTRAS: tuple[str, ...] = ("export", "clearml", "sahi", "dev")

UBUNTU_WEASYPRINT_APT_HINT = (
    "sudo apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info"
)


@dataclass(frozen=True)
class DepCheckRow:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ExportDepsReport:
    rows: tuple[DepCheckRow, ...]
    export_ready: bool

    @property
    def pandoc_ok(self) -> bool:
        for row in self.rows:
            if row.name == "pandoc":
                return row.ok
        return False


def known_optional_extras() -> tuple[str, ...]:
    return KNOWN_OPTIONAL_EXTRAS


def _project_root() -> Path | None:
    pkg_dir = Path(smartrain.__file__).resolve().parent
    root = pkg_dir.parent
    if (root / "pyproject.toml").is_file():
        return root
    return None


def build_install_spec(extras: Iterable[str]) -> str:
    normalized = [str(x).strip() for x in extras if str(x).strip()]
    unknown = sorted(set(normalized) - set(KNOWN_OPTIONAL_EXTRAS))
    if unknown:
        raise ValueError(f"Unknown extras: {', '.join(unknown)}")
    extra_part = ",".join(normalized)
    root = _project_root()
    if root is not None:
        base = str(root)
        return f"{base}[{extra_part}]" if extra_part else base
    return f"{PACKAGE_NAME}[{extra_part}]" if extra_part else PACKAGE_NAME


def _import_ok(module_name: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return False, str(exc)
    if spec is None:
        return False, "not installed"
    return True, "import ok"


def check_export_deps() -> ExportDepsReport:
    pypandoc_ok, pypandoc_detail = _import_ok("pypandoc")
    pandoc_ok, pandoc_detail = check_pandoc_ready(quiet=True)
    weasy_ok, weasy_detail = check_weasyprint_ready()
    fpdf_ok, fpdf_detail = _import_ok("fpdf")
    odf_ok, odf_detail = _import_ok("odf")

    rows = (
        DepCheckRow("pypandoc (base)", pypandoc_ok, pypandoc_detail),
        DepCheckRow("pandoc", pandoc_ok, pandoc_detail),
        DepCheckRow("weasyprint (export extra)", weasy_ok, weasy_detail),
        DepCheckRow("fpdf2 (base)", fpdf_ok, fpdf_detail),
        DepCheckRow("odfpy (base)", odf_ok, odf_detail),
    )
    return ExportDepsReport(rows=rows, export_ready=pandoc_ok)


def install_optional_extras(extras: Iterable[str], *, dry_run: bool = False) -> str:
    spec = build_install_spec(extras)
    cmd = [sys.executable, "-m", "pip", "install", spec]
    if dry_run:
        return " ".join(cmd)
    subprocess.check_call(cmd)
    return " ".join(cmd)


def ubuntu_weasyprint_apt_hint() -> str | None:
    try:
        if Path("/etc/os-release").is_file():
            text = Path("/etc/os-release").read_text(encoding="utf-8").lower()
            if "ubuntu" in text or "debian" in text:
                return UBUNTU_WEASYPRINT_APT_HINT
    except Exception:
        return None
    return None
