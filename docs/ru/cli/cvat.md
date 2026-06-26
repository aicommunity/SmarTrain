> English version: [../../cli/cvat.md](../../cli/cvat.md)

# CLI: CVAT

`smartrain cvat` предоставляет конвертацию между CVAT for images 1.1, YOLO и CvsDclDet.

## Команды

```bash
smartrain cvat import --cvat-zip /path/to/export.zip --output-dir /path/to/yolo_dataset
smartrain cvat export --dataset-dir /path/to/yolo_dataset --zip-path /path/to/out.cvat11.zip
smartrain cvat from-cvsdcldet
smartrain cvat from-cvsdcldet --source-dir raw_data/my_det --output-dir converted_raw_data/my_det --zip
smartrain cvat from-cvsdcldet --source-dir raw_data/my_det --rename-classes white_line line
```

### `from-cvsdcldet`

Конвертирует папку **CvsDclDet** (пары `*.jpg` + `*.json` с bbox в пикселях) в layout **CVAT for images 1.1** (`annotations.xml` + `images/`).

- **Интерактивный режим** (без флагов, TTY): запрашивает источник (из `raw_data/` или путь вручную), приёмник (по умолчанию `converted_raw_data/<имя>/`), опциональное переименование классов и опциональный ZIP.
- **`--rename-classes OLD NEW`**: повторяемый флаг переименования классов (например `--rename-classes white_line line`).
- **`--zip`**: дополнительно создать `<output-dir>.cvat11.zip` для импорта в CVAT.
- **`--force`**: перезаписать существующий результат.

Также `fusion` умеет напрямую работать с источниками CVAT for images 1.1 через временную генерацию YOLO-меток.
В метаданных SmarTrain layout CVAT хранится под внутренним ID `cvat11`; исходники CvsDclDet — под `cvsdcldet`.
