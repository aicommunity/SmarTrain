from __future__ import annotations

from pathlib import Path

from PIL import Image

from smartrain.workflows.datasets.datasets_json_former import main as datasets_json_main
from smartrain.workflows.datasets.dataset_former import main as dataset_former_main
from smartrain.core.runtime.workspace_paths import deploy_workspace


def _write_jpg(path: Path, *, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color=(1, 2, 3))
    im.save(path, format="JPEG", quality=85)


def test_dataset_former_supports_native_cvat11(tmp_path: Path) -> None:
    deploy_workspace(str(tmp_path))

    # Create CVAT extracted dataset inside raw_data without relying on folder name.
    ds_root = tmp_path / "raw_data" / "any_name_here"
    nested = ds_root / "export_payload"
    images_dir = nested / "images"
    _write_jpg(images_dir / "img001.jpg", size=(100, 80))

    (nested / "annotations.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>taskX</name>
      <labels>
        <label><name>cat</name><type>bbox</type><attributes></attributes></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="img001.jpg" width="100" height="80">
    <box label="cat" xtl="10" ytl="10" xbr="60" ybr="50" occluded="0" z_order="0"></box>
  </image>
</annotations>
""",
        encoding="utf-8",
    )

    # 1) Scan datasets -> datasets_info.json includes structure=cvat11 and classes.
    datasets_json_main(["--workspace", str(tmp_path)])

    # 2) Build merged dataset from CVAT source directly.
    dataset_former_main(
        [
            "--workspace",
            str(tmp_path),
            "--output-name",
            "merged",
            "--dataset",
            "any_name_here",
            "--classes",
            "cat",
        ]
    )

    out_root = tmp_path / "datasets" / "merged"
    assert (out_root / "data.yaml").exists()

    # At least one image/label pair must be produced (may land in test split for n=1).
    found_img = list(out_root.glob("*/images/*.jpg"))
    found_lbl = list(out_root.glob("*/labels/*.txt"))
    assert found_img, "expected at least one output image"
    assert found_lbl, "expected at least one output label"

