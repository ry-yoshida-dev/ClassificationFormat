import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .label_annotation import LabelAnnotation
from .label_presence import LabelPresence


@dataclass(frozen=True, eq=False)
class ImageAnnotation:
    """
    Class decisions made for one image.

    Attributes
    ----------
    image_path : Path
        Location of the image file.
    labels : tuple[LabelAnnotation, ...], optional
        At most one decision per class; classes without a label are not annotated.

    Raises
    ------
    ValueError
        If a class is labeled more than once.
    """

    image_path: Path
    labels: tuple[LabelAnnotation, ...] = ()

    def __post_init__(self) -> None:
        class_name_counts: Counter[str] = Counter(label.class_name for label in self.labels)
        repeated_class_names: list[str] = [name for name, count in class_name_counts.items() if count > 1]
        if repeated_class_names:
            raise ValueError(f"{self.image_path} labels classes more than once: {repeated_class_names}")

    @classmethod
    def from_scores(cls, image_path: Path, scores: Mapping[str, float], threshold: float) -> "ImageAnnotation":
        """
        Decide every scored class by a score threshold, keeping the scores as confidences.

        Parameters
        ----------
        image_path : Path
            Location of the image file.
        scores : Mapping[str, float]
            Classifier score in ``[0, 1]`` of each class, e.g. sigmoid outputs of a multi-label classifier.
        threshold : float
            Lowest score in ``[0, 1]`` marking a class as present; lower scores mark it absent.

        Returns
        -------
        ImageAnnotation
            Image with one present or absent label per scored class, in mapping order.

        Raises
        ------
        ValueError
            If the threshold or a score is outside ``[0, 1]``.
        """
        if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1]. got {threshold}")
        labels: tuple[LabelAnnotation, ...] = tuple(
            LabelAnnotation(
                class_name=class_name,
                presence=LabelPresence.PRESENT if score >= threshold else LabelPresence.ABSENT,
                confidence=score,
            )
            for class_name, score in scores.items()
        )
        return cls(image_path=image_path, labels=labels)

    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Classes labeled in this image, whatever their presence.

        Returns
        -------
        tuple[str, ...]
            Class names in label order.
        """
        return tuple(label.class_name for label in self.labels)

    @property
    def present_class_names(self) -> tuple[str, ...]:
        """
        Classes shown in this image.

        Returns
        -------
        tuple[str, ...]
            Class names of present labels in label order.
        """
        return tuple(label.class_name for label in self.labels if label.is_present)

    def find_label(self, class_name: str) -> LabelAnnotation | None:
        """
        Decision about a class.

        Parameters
        ----------
        class_name : str
            Class to look up.

        Returns
        -------
        LabelAnnotation | None
            The label of the class, or ``None`` if the class is not annotated in this image.
        """
        return self._labels_by_class_name.get(class_name)

    @cached_property
    def _labels_by_class_name(self) -> dict[str, LabelAnnotation]:
        return {label.class_name: label for label in self.labels}
