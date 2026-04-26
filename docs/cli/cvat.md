> Russian version: [../ru/cli/cvat.md](../ru/cli/cvat.md)

# CLI: CVAT

`smartrain cvat` provides conversion between CVAT for images 1.1 and YOLO.

## Commands

```bash
smartrain cvat import --cvat-zip /path/to/export.zip --output-dir /path/to/yolo_dataset
smartrain cvat export --dataset-dir /path/to/yolo_dataset --zip-path /path/to/out.cvat11.zip
```

Also, `fusion` can work directly with CVAT for images 1.1 sources via temporary YOLO label generation.
In SmarTrain metadata this format is represented by internal structure ID `cvat11`.
