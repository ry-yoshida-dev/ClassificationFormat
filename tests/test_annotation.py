from dataclasses import replace
from pathlib import Path

import pytest

from classification_format import (
    AnnotationDataset,
    ImageAnnotation,
    LabelAnnotation,
    LabelCardinality,
    LabelPresence,
)


def test_confidence_outside_unit_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        LabelAnnotation(class_name="cat", confidence=1.5)


def test_blank_class_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="blank"):
        LabelAnnotation(class_name=" ")


def test_class_name_with_line_break_is_rejected() -> None:
    with pytest.raises(ValueError, match="line breaks"):
        LabelAnnotation(class_name="cat\ndog")


def test_image_rejects_repeated_class() -> None:
    with pytest.raises(ValueError, match="more than once"):
        ImageAnnotation(
            image_path=Path("a.jpg"),
            labels=(
                LabelAnnotation(class_name="cat"),
                LabelAnnotation(class_name="cat", presence=LabelPresence.ABSENT),
            ),
        )


def test_image_separates_present_and_listed_classes(multi_label_dataset: AnnotationDataset) -> None:
    street: ImageAnnotation = multi_label_dataset.images[0]
    assert street.class_names == ("person", "car", "dog")
    assert street.present_class_names == ("person",)
    assert street.find_label("car") == LabelAnnotation(class_name="car", presence=LabelPresence.ABSENT, confidence=0.25)
    assert street.find_label("bird") is None


def test_from_scores_thresholds_scores() -> None:
    image: ImageAnnotation = ImageAnnotation.from_scores(Path("a.jpg"), {"cat": 0.7, "dog": 0.3}, threshold=0.5)
    assert image.present_class_names == ("cat",)
    assert image.labels[1] == LabelAnnotation(class_name="dog", presence=LabelPresence.ABSENT, confidence=0.3)


def test_from_scores_rejects_threshold_outside_unit_range() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ImageAnnotation.from_scores(Path("a.jpg"), {"cat": 0.7}, threshold=-0.1)


def test_dataset_orders_given_classes_first(multi_label_dataset: AnnotationDataset) -> None:
    assert multi_label_dataset.class_names == ("person", "car", "dog")
    assert multi_label_dataset.class_index("dog") == 2


def test_dataset_rejects_unlisted_class(multi_label_dataset: AnnotationDataset) -> None:
    with pytest.raises(ValueError, match="missing from class_names"):
        AnnotationDataset(images=multi_label_dataset.images, class_names=("person",))


def test_dataset_rejects_duplicate_image_paths(multi_label_dataset: AnnotationDataset) -> None:
    image: ImageAnnotation = multi_label_dataset.images[0]
    with pytest.raises(ValueError, match="image paths"):
        AnnotationDataset(images=(image, image), class_names=multi_label_dataset.class_names)


def test_dataset_rejects_differently_spelled_duplicate_image_paths(multi_label_dataset: AnnotationDataset) -> None:
    image: ImageAnnotation = multi_label_dataset.images[0]
    detour_path: Path = image.image_path.parent / "detour" / ".." / image.image_path.name
    with pytest.raises(ValueError, match="image paths"):
        AnnotationDataset(
            images=(image, replace(image, image_path=detour_path)), class_names=multi_label_dataset.class_names
        )


def test_single_label_dataset_needs_one_present_label(multi_label_dataset: AnnotationDataset) -> None:
    with pytest.raises(ValueError, match="exactly one present label"):
        AnnotationDataset(
            images=multi_label_dataset.images,
            class_names=multi_label_dataset.class_names,
            label_cardinality=LabelCardinality.SINGLE,
        )


def test_dataset_rejects_unknown_class_lookup(multi_label_dataset: AnnotationDataset) -> None:
    with pytest.raises(KeyError, match="unknown class"):
        multi_label_dataset.class_index("bird")
