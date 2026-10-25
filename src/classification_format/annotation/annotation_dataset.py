from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .image_annotation import ImageAnnotation
from .label_cardinality import LabelCardinality


@dataclass(frozen=True, eq=False)
class AnnotationDataset:
    """
    Annotated images sharing one class list.

    The order of ``class_names`` defines class ids and the column order of tabular formats.

    Attributes
    ----------
    images : tuple[ImageAnnotation, ...]
        Annotated images.
    class_names : tuple[str, ...]
        Every class that may appear, in id order.
    label_cardinality : LabelCardinality, optional
        Number of present classes each image may have.

    Raises
    ------
    ValueError
        If class names repeat, two image paths resolve to the same file, an image labels an unlisted class, or a
        single-label image does not have exactly one present label.
    """

    images: tuple[ImageAnnotation, ...]
    class_names: tuple[str, ...]
    label_cardinality: LabelCardinality = LabelCardinality.MULTI

    def __post_init__(self) -> None:
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(f"class_names must be unique. got {self.class_names}")
        image_paths: list[Path] = [image.image_path.resolve() for image in self.images]
        if len(set(image_paths)) != len(image_paths):
            raise ValueError("image paths must be unique")
        known_class_names: set[str] = set(self.class_names)
        for image in self.images:
            unknown_class_names: list[str] = [name for name in image.class_names if name not in known_class_names]
            if unknown_class_names:
                raise ValueError(f"{image.image_path} uses classes missing from class_names: {unknown_class_names}")
        match self.label_cardinality:
            case LabelCardinality.SINGLE:
                for image in self.images:
                    present_class_count: int = len(image.present_class_names)
                    if present_class_count != 1:
                        raise ValueError(
                            f"single-label images need exactly one present label. {image.image_path} has "
                            + f"{present_class_count}"
                        )
            case LabelCardinality.MULTI:
                pass

    @classmethod
    def from_images(
        cls,
        images: Iterable[ImageAnnotation],
        class_names: Iterable[str] = (),
        label_cardinality: LabelCardinality = LabelCardinality.MULTI,
    ) -> "AnnotationDataset":
        """
        Build a dataset whose class list covers every label.

        Parameters
        ----------
        images : Iterable[ImageAnnotation]
            Annotated images.
        class_names : Iterable[str], optional
            Classes placed first, in this order, even if no label uses them.
        label_cardinality : LabelCardinality, optional
            Number of present classes each image may have.

        Returns
        -------
        AnnotationDataset
            Dataset with ``class_names`` followed by the remaining classes in order of first appearance.
        """
        image_tuple: tuple[ImageAnnotation, ...] = tuple(images)
        ordered_class_names: dict[str, None] = dict.fromkeys(class_names)
        for image in image_tuple:
            ordered_class_names.update(dict.fromkeys(image.class_names))
        return cls(images=image_tuple, class_names=tuple(ordered_class_names), label_cardinality=label_cardinality)

    def class_index(self, class_name: str) -> int:
        """
        Zero-based position of a class.

        Parameters
        ----------
        class_name : str
            Listed class.

        Returns
        -------
        int
            Index into ``class_names``.

        Raises
        ------
        KeyError
            If the class is not listed.
        """
        class_indices: dict[str, int] = self._class_indices
        if class_name not in class_indices:
            raise KeyError(f"unknown class: {class_name!r}")
        return class_indices[class_name]

    @cached_property
    def _class_indices(self) -> dict[str, int]:
        return {class_name: index for index, class_name in enumerate(self.class_names)}
