# CLI: CVAT

`smartrain cvat` предоставляет конвертацию между CVAT 1.1 и YOLO.

## Команды

```bash
smartrain cvat import --cvat-zip /path/to/export.zip --output-dir /path/to/yolo_dataset
smartrain cvat export --dataset-dir /path/to/yolo_dataset --zip-path /path/to/out.cvat11.zip
```

Также `fusion` умеет работать с `structure="cvat11"` напрямую через временную генерацию YOLO-меток.
