> Russian version: [../ru/cli/inference-tools.md](../ru/cli/inference-tools.md)

# CLI: inference tools

## `sahi`

Tile inference for large images.

```bash
smartrain sahi --model /path/to/model.pt --source /path/to/image_or_dir --output sahi_out
smartrain sahi --model /path/to/model.pt --source images/ --slice-h 768 --slice-w 768 --overlap-h 0.25 --overlap-w 0.25
```

Requires additional dependency: `pip install -e ".[sahi]"`.

Output: tiled prediction artifacts in the specified `--output` directory.

## `heatmap`

Generating a heat map from an image:

```bash
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --output heatmap.png
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --colormap 12
```

Output: single heatmap image file (PNG/JPG depending on target path extension).
