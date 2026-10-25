# annotation

## Overview

Immutable, validated annotation model shared by every writer. Each image holds at most one `LabelAnnotation` per
class, recording whether the class is present, absent or ambiguous and an optional confidence; classes without a
label are not annotated. Invalid data (repeated classes, unknown classes, single-label images without exactly one
present class) is rejected at construction.

## Components

| Component | Description |
| --------- | ----------- |
| [label_presence.py](./label_presence.py) | `LabelPresence`: present, absent or ambiguous. |
| [label_cardinality.py](./label_cardinality.py) | `LabelCardinality`: single-label or multi-label dataset. |
| [label_annotation.py](./label_annotation.py) | `LabelAnnotation`: class name, presence and optional confidence. |
| [image_annotation.py](./image_annotation.py) | `ImageAnnotation`: image path and its labels, built directly or from thresholded scores. |
| [annotation_dataset.py](./annotation_dataset.py) | `AnnotationDataset`: images, the ordered class list and the label cardinality. |

## Examples

```python
image = ImageAnnotation.from_scores(Path("a.jpg"), {"cat": 0.7, "dog": 0.3}, threshold=0.5)
image.present_class_names
dataset = AnnotationDataset.from_images([image], class_names=["cat", "dog", "bird"])
dataset.class_index("dog")
```
