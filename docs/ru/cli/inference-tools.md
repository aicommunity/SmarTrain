> English version: [../../cli/inference-tools.md](../../cli/inference-tools.md)

# CLI: инструменты инференса

## `sahi`

Тайловый инференс для больших изображений.

```bash
smartrain sahi --model /path/to/best.pt --source /path/to/image_or_dir --output sahi_out
```

Требует дополнительной зависимости: `pip install -e ".[sahi]"`.

## `heatmap`

Генерация тепловой карты по изображению:

```bash
smartrain heatmap --model /path/to/best.pt --source /path/to/image.jpg --output heatmap.png
```
