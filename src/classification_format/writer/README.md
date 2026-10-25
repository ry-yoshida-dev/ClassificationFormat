# writer

## Overview

Writers turning an `AnnotationDataset` into files. Every writer implements
`AnnotationWriter.write(dataset, output_directory)`, creates the directory if needed and returns the written paths.
Writers validate file name collisions and missing images before writing anything.

## Components

| Component | Description |
| --------- | ----------- |
| [base.py](./base.py) | `AnnotationWriter`: abstract writer with shared validation. |
| [image_transfer.py](./image_transfer.py) | `ImageTransfer`: copy, symbolic link or hard link of image files. |
| [formats/](./formats/README.md) | Concrete writers for ImageFolder, CSV, Hugging Face, Pascal VOC and Open Images. |
| [annotation_format.py](./annotation_format.py) | `AnnotationFormat`: supported formats, display names, multi-label support and default writers. |
| [json_value.py](./json_value.py) | `JsonValue`: recursive type of JSON documents. |

## Examples

```python
for annotation_format in AnnotationFormat:
    if annotation_format.is_multi_label_supported:
        annotation_format.create_writer().write(dataset, Path("export") / annotation_format.value)
```
