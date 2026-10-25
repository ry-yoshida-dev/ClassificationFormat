# formats

## Overview

Concrete writers, one per classification format. Writers that place images (ImageFolder, Hugging Face) copy or
link them with an `ImageTransfer`; the others reference images by path or stem.

## Components

| Component | Description |
| --------- | ----------- |
| [image_folder.py](./image_folder.py) | `ImageFolderWriter`: one directory per class, single-label only. |
| [csv_layout.py](./csv_layout.py) | `CsvLayout`: label list, binary or score columns. |
| [csv_table.py](./csv_table.py) | `CsvWriter`: one CSV row per image in the chosen layout. |
| [hugging_face.py](./hugging_face.py) | `HuggingFaceWriter`: Hugging Face `imagefolder` with `metadata.jsonl`. |
| [pascal_voc.py](./pascal_voc.py) | `PascalVocWriter`: Pascal VOC classification image sets. |
| [open_images.py](./open_images.py) | `OpenImagesWriter`: Open Images image-level label CSV. |

## Examples

```python
CsvWriter(CsvLayout.BINARY, is_unlisted_absent=False).write(dataset, Path("export/csv"))
HuggingFaceWriter(ImageTransfer.HARDLINK, is_confidence_included=True).write(dataset, Path("export/hf"))
```
