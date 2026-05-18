> English version: [../../cli/inference-tools.md](../../cli/inference-tools.md)

# CLI: инструменты инференса

## `sahi`

Тайловый инференс для больших изображений.

```bash
smartrain sahi --model /path/to/model.pt --source /path/to/image_or_dir --output sahi_out
smartrain sahi --model /path/to/model.pt --source images/ --slice-h 768 --slice-w 768 --overlap-h 0.25 --overlap-w 0.25
```

Требует дополнительной зависимости: `pip install -e ".[sahi]"`.

Результат: тайловые предсказания в каталоге, указанном через `--output`.

## `heatmap`

Генерация тепловой карты по изображению:

```bash
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --output heatmap.png
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --colormap 12
```

Результат: один файл heatmap (PNG/JPG в зависимости от расширения пути назначения).
