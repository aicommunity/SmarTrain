from __future__ import annotations

from pathlib import Path

from PIL import Image

from smartrain.services.datasets.yolo_labels import YoloBBox
from smartrain.services.visualization.color_registry import LabelColorRegistry
from smartrain.services.visualization.rendering import (
    _annotation_scale,
    _is_grayscale_image,
    _vis_draw_metrics,
    render_combined_overlay,
    save_rendered_image,
)


def test_annotation_scale_only_grows_for_images_larger_than_1080p() -> None:
    assert _annotation_scale(960, 540) == 1.0
    assert _annotation_scale(1920, 1080) == 1.0
    assert _annotation_scale(3840, 2160) == 2.0


def test_vis_draw_metrics_use_large_base_font(tmp_path: Path) -> None:
    metrics = _vis_draw_metrics(128, 96)
    assert metrics.font_size >= 20
    assert metrics.line_width >= 3

    large = _vis_draw_metrics(3840, 2160)
    assert large.font_size > metrics.font_size
    assert large.line_width > metrics.line_width


def test_grayscale_image_is_detected_and_colorized_on_render(tmp_path: Path) -> None:
    gray = tmp_path / "gray.png"
    Image.new("L", (120, 90), color=128).save(gray)
    with Image.open(gray) as im:
        assert _is_grayscale_image(im)

    rendered, fmt = render_combined_overlay(
        gray,
        [YoloBBox(cls_id=0, cx=0.5, cy=0.5, w=0.4, h=0.4)],
        [],
        {0: "obj"},
    )
    assert rendered.mode in {"RGB", "RGBA"}
    assert fmt == "PNG"
    out = tmp_path / "out.png"
    save_rendered_image(rendered, out, original_format=fmt)
    with Image.open(out) as saved:
        assert saved.format == "PNG"
        assert saved.mode in {"RGB", "RGBA", "P"}


def test_label_color_registry_is_stable_and_persistent(tmp_path: Path) -> None:
    reg = LabelColorRegistry(tmp_path)
    c1 = reg.ensure("cat")
    reg.save()
    p = tmp_path / "label_colors.json"
    assert p.is_file()
    reg2 = LabelColorRegistry(tmp_path)
    c2 = reg2.ensure("cat")
    assert c1 == c2
    reg2.ensure("dog")
    reg2.save()
    payload = p.read_text(encoding="utf-8")
    assert "cat" in payload and "dog" in payload
