> English version: [../../cli/cvat.md](../../cli/cvat.md)

# CLI: CVAT

`smartrain cvat` предоставляет конвертацию между CVAT for images 1.1 и YOLO.

## Команды

```bash
smartrain cvat import --cvat-zip /path/to/export.zip --output-dir /path/to/yolo_dataset
smartrain cvat export --dataset-dir /path/to/yolo_dataset --zip-path /path/to/out.cvat11.zip
```

Также `fusion` умеет напрямую работать с источниками CVAT for images 1.1 через временную генерацию YOLO-меток.
В метаданных SmarTrain этот формат хранится под внутренним идентификатором структуры `cvat11`.
