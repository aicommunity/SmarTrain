> Russian version: [../ru/cli/inference-tools.md](../ru/cli/inference-tools.md)

# CLI: inference tools

## `sahi`

Tile inference for large images.

```bash
smartrain sahi --model /path/to/best.pt --source /path/to/image_or_dir --output sahi_out
```

Requires additional dependency: `pip install -e ".[sahi]"`.

## `heatmap`

Generating a heat map from an image:

```bash
smartrain heatmap --model /path/to/best.pt --source /path/to/image.jpg --output heatmap.png
```
