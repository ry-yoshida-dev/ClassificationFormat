# ClassificationFormat

## Overview

Export single-label and multi-label image classification annotations to common dataset formats. The library does not
depend on any classifier, and has no runtime dependencies: build an `AnnotationDataset` from your own labels and
pick a writer. It is the classification counterpart of
[ObjectDetectionFormat](https://github.com/ry-yoshida-dev/ObjectDetectionFormat) and
[SegmentationFormat](https://github.com/ry-yoshida-dev/SegmentationFormat).

Multi-label data is modeled explicitly:

- **Three label states**: each `LabelAnnotation` is `PRESENT`, `ABSENT` or `AMBIGUOUS` (Pascal VOC `difficult`,
  CheXpert uncertain). A class with no label in an image is not annotated; each writer maps that to its own
  convention.
- **Cardinality is declared**: `LabelCardinality.SINGLE` datasets need exactly one present class per image, while
  `MULTI` datasets allow any number. Writers that store one class per image reject multi-label datasets.
- **Scores are kept**: every label may carry a classifier confidence, and `ImageAnnotation.from_scores` turns
  per-class scores (e.g. SigLIP sigmoids) into present and absent labels with a threshold.

| Format | Output | Multi-label | Label encoding |
| ------ | ------ | :---------: | -------------- |
| ImageFolder | `<class>/<image>` (copied or linked) | - | directory name |
| CSV | `labels.csv` | yes | `1 / 0 / -1` per class, confidence per class, or `cat\|dog` label list |
| Hugging Face imagefolder | images + `metadata.jsonl` | yes | `label` or `labels`, optional `scores` |
| Pascal VOC | `ImageSets/Main/<split>.txt`, `<class>_<split>.txt` | yes | `1 / -1 / 0` (present / absent / difficult) |
| Open Images | `image_labels.csv` | yes | `ImageID,Source,LabelName,Confidence`, human `1 / 0` or machine scores |

See [src/classification_format](src/classification_format/README.md) for the package layout.

## Installation

```bash
pip install git+https://github.com/ry-yoshida-dev/ClassificationFormat.git
pip install -e ".[dev]"
```

## Examples

```python
from pathlib import Path

from classification_format import (
    AnnotationDataset,
    AnnotationFormat,
    CsvLayout,
    CsvWriter,
    ImageAnnotation,
    LabelAnnotation,
    LabelPresence,
    PascalVocWriter,
)

image = ImageAnnotation(
    image_path=Path("images/street.jpg"),
    labels=(
        LabelAnnotation(class_name="person"),
        LabelAnnotation(class_name="car", presence=LabelPresence.ABSENT),
        LabelAnnotation(class_name="dog", presence=LabelPresence.AMBIGUOUS),
    ),
)
dataset = AnnotationDataset.from_images([image], class_names=["person", "car", "dog"])

AnnotationFormat.OPEN_IMAGES.create_writer().write(dataset, Path("export/open_images"))
CsvWriter(CsvLayout.BINARY).write(dataset, Path("export/csv"))
PascalVocWriter(split_name="train").write(dataset, Path("export/VOC"))
```

Single-label datasets can also be exported as class directories:

```python
from classification_format import ImageFolderWriter, ImageTransfer, LabelCardinality

dataset = AnnotationDataset.from_images(images, label_cardinality=LabelCardinality.SINGLE)
ImageFolderWriter(ImageTransfer.SYMLINK).write(dataset, Path("export/train"))
```

Converting an [OpenVocabularyClassification](https://github.com/ry-yoshida-dev/OpenVocabularyClassification) result,
keeping every score:

```python
image = ImageAnnotation.from_scores(
    path, {classification.class_name: classification.score for classification in result}, threshold=0.1
)
CsvWriter(CsvLayout.SCORE).write(AnnotationDataset.from_images([image]), Path("export/scores"))
```
