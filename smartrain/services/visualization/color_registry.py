from __future__ import annotations

import colorsys
import hashlib
import json
from pathlib import Path


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    s = str(value).strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {value!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _deterministic_color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    sat = 0.88
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


class LabelColorRegistry:
    def __init__(self, workspace_root: Path, file_name: str = "label_colors.json") -> None:
        self.path = Path(workspace_root) / file_name
        self._colors: dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        for k, v in payload.items():
            try:
                rgb = _hex_to_rgb(str(v))
            except Exception:
                continue
            self._colors[str(k)] = _rgb_to_hex(rgb)

    def ensure(self, label: str) -> tuple[int, int, int]:
        key = str(label).strip()
        if not key:
            key = "unknown"
        if key not in self._colors:
            self._colors[key] = _rgb_to_hex(_deterministic_color_for_label(key))
            self._dirty = True
        return _hex_to_rgb(self._colors[key])

    def colors_rgb(self) -> dict[str, tuple[int, int, int]]:
        return {k: _hex_to_rgb(v) for k, v in self._colors.items()}

    def save(self) -> None:
        if not self._dirty and self.path.is_file():
            return
        self.path.write_text(
            json.dumps(dict(sorted(self._colors.items())), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._dirty = False

