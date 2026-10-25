from pathlib import Path

import pytest

from classification_format import (
    AnnotationDataset,
    ImageAnnotation,
    LabelAnnotation,
    LabelCardinality,
    LabelPresence,
)


def create_image(path: Path) -> Path:
    """
    Create a placeholder image file.

    Parameters
    ----------
    path : Path
        Location of the file.

    Returns
    -------
    Path
        The created file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())
    return path


@pytest.fixture
def multi_label_dataset(tmp_path: Path) -> AnnotationDataset:
    """
    Multi-label dataset covering every label presence, scored and unscored labels, and an unlabeled image.

    Parameters
    ----------
    tmp_path : Path
        Directory receiving the placeholder images.

    Returns
    -------
    AnnotationDataset
        Images ``street``, ``park`` and ``empty`` with classes ``person``, ``car`` and ``dog``.
    """
    street: ImageAnnotation = ImageAnnotation(
        image_path=create_image(tmp_path / "images" / "street.jpg"),
        labels=(
            LabelAnnotation(class_name="person", confidence=0.9),
            LabelAnnotation(class_name="car", presence=LabelPresence.ABSENT, confidence=0.25),
            LabelAnnotation(class_name="dog", presence=LabelPresence.AMBIGUOUS),
        ),
    )
    park: ImageAnnotation = ImageAnnotation(
        image_path=create_image(tmp_path / "images" / "park.png"),
        labels=(LabelAnnotation(class_name="dog"), LabelAnnotation(class_name="person")),
    )
    empty: ImageAnnotation = ImageAnnotation(image_path=create_image(tmp_path / "images" / "empty.png"))
    return AnnotationDataset.from_images([street, park, empty], class_names=["person", "car"])


@pytest.fixture
def single_label_dataset(tmp_path: Path) -> AnnotationDataset:
    """
    Single-label dataset with an unused class.

    Parameters
    ----------
    tmp_path : Path
        Directory receiving the placeholder images.

    Returns
    -------
    AnnotationDataset
        Images ``cat`` and ``dog`` with classes ``cat``, ``dog`` and ``bird``.
    """
    cat: ImageAnnotation = ImageAnnotation(
        image_path=create_image(tmp_path / "images" / "cat.jpg"),
        labels=(
            LabelAnnotation(class_name="cat", confidence=0.8),
            LabelAnnotation(class_name="dog", presence=LabelPresence.ABSENT, confidence=0.2),
        ),
    )
    dog: ImageAnnotation = ImageAnnotation(
        image_path=create_image(tmp_path / "images" / "dog.jpg"), labels=(LabelAnnotation(class_name="dog"),)
    )
    return AnnotationDataset.from_images(
        [cat, dog], class_names=["cat", "dog", "bird"], label_cardinality=LabelCardinality.SINGLE
    )
