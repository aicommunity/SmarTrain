> Russian version: [../ru/cli/inference-tools.md](../ru/cli/inference-tools.md)

# CLI: inference tools

## `sahi`

Tile inference for large images and slicing-aided fine-tune dataset prep ([arXiv:2202.06934](https://arxiv.org/abs/2202.06934)).

### `infer` (default / legacy)

```bash
smartrain sahi infer --model /path/to/model.pt --source /path/to/image_or_dir --output sahi_out
smartrain sahi --model /path/to/model.pt --source images/ --slice-h 768 --slice-w 768 --overlap-h 0.25 --overlap-w 0.25
```

Requires: `pip install -e ".[sahi]"`.

### `prepare-slices` (fine-tune recipe)

Build a new YOLO dataset of sliding-window crops (defaults match infer: slice 640, overlap 0.2):

```bash
smartrain sahi prepare-slices --workspace /path/to/ws --dataset my_ds
# → datasets/my_ds_sahi_slices (+ data.yaml, passport, sidecar)
smartrain train --workspace /path/to/ws --dataset my_ds_sahi_slices ...
smartrain sahi infer --model runs/.../best.pt --source large_images/
```

`prepare-slices` does **not** require the `sahi` package.

## `heatmap`

Generating a heat map from an image:

```bash
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --output heatmap.png
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --colormap 12
```

Output: single heatmap image file (PNG/JPG depending on target path extension).
