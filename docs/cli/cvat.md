> Russian version: [../ru/cli/cvat.md](../ru/cli/cvat.md)

# CLI: CVAT

`smartrain cvat` provides conversion between CVAT for images 1.1, YOLO, and CvsDclDet.

## Commands

```bash
smartrain cvat import --cvat-zip /path/to/export.zip --output-dir /path/to/yolo_dataset
smartrain cvat export --dataset-dir /path/to/yolo_dataset --zip-path /path/to/out.cvat11.zip
smartrain cvat from-cvsdcldet
smartrain cvat from-cvsdcldet --source-dir raw_data/my_det --output-dir converted_raw_data/my_det --zip
smartrain cvat from-cvsdcldet --source-dir raw_data/my_det --rename-classes white_line line
```

### `from-cvsdcldet`

Converts a **CvsDclDet** folder (paired `*.jpg` + `*.json` with pixel bbox detections) into **CVAT for images 1.1** layout (`annotations.xml` + `images/`).

- **Interactive mode** (empty flags, TTY): prompts for source (from `raw_data/` or manual path), output (`converted_raw_data/<name>/` by default), optional class rename, and optional ZIP.
- **`--rename-classes OLD NEW`**: repeat to rename class labels in the output (e.g. `--rename-classes white_line line`).
- **`--zip`**: also write `<output-dir>.cvat11.zip` importable into CVAT.
- **`--force`**: overwrite existing output.

Also, `fusion` can work directly with CVAT for images 1.1 sources via temporary YOLO label generation.
In SmarTrain metadata CVAT layout is represented by internal structure ID `cvat11`; CvsDclDet sources use `cvsdcldet`.
