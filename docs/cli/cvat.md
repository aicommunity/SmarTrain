> Russian version: [../ru/cli/cvat.md](../ru/cli/cvat.md)

# CLI: CVAT

`smartrain cvat` provides conversion between CVAT 1.1 and YOLO.

## Commands

```bash
smartrain cvat import --cvat-zip /path/to/export.zip --output-dir /path/to/yolo_dataset
smartrain cvat export --dataset-dir /path/to/yolo_dataset --zip-path /path/to/out.cvat11.zip
```

Also, `fusion` can work with `structure="cvat11"` directly through the temporary generation of YOLO tags.
