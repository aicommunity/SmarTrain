> English version: [../../cli/inference-tools.md](../../cli/inference-tools.md)

# CLI: инструменты инференса

## `sahi`

Тайловый инференс и подготовка датасета для slicing-aided fine-tune ([arXiv:2202.06934](https://arxiv.org/abs/2202.06934)).

### `infer` (по умолчанию / legacy)

```bash
smartrain sahi infer --model /path/to/model.pt --source /path/to/image_or_dir --output sahi_out
smartrain sahi --model /path/to/model.pt --source images/ --slice-h 768 --slice-w 768 --overlap-h 0.25 --overlap-w 0.25
```

Требует: `pip install -e ".[sahi]"`.

### `prepare-slices` (рецепт fine-tune)

Новый YOLO-датасет из sliding-window кропов (defaults как у infer: slice 640, overlap 0.2):

```bash
smartrain sahi prepare-slices --workspace /path/to/ws --dataset my_ds
# → datasets/my_ds_sahi_slices (+ data.yaml, passport, sidecar)
smartrain train --workspace /path/to/ws --dataset my_ds_sahi_slices ...
smartrain sahi infer --model runs/.../best.pt --source large_images/
```

`prepare-slices` **не** требует пакет `sahi`.

## `heatmap`

Генерация тепловой карты по изображению:

```bash
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --output heatmap.png
smartrain heatmap --model /path/to/model.pt --source /path/to/image.jpg --colormap 12
```

Результат: один файл heatmap (PNG/JPG в зависимости от расширения пути назначения).
