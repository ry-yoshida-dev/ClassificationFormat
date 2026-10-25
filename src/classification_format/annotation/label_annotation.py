import math
from dataclasses import dataclass

from .label_presence import LabelPresence


@dataclass(frozen=True)
class LabelAnnotation:
    """
    Decision about one class in one image.

    Attributes
    ----------
    class_name : str
        Category the decision refers to.
    presence : LabelPresence, optional
        Whether the class is shown.
    confidence : float | None, optional
        Classifier score in ``[0, 1]`` that the class is shown; ``None`` for ground-truth labels.

    Raises
    ------
    ValueError
        If the class name is blank or contains a line break, or the confidence is outside ``[0, 1]``.
    """

    class_name: str
    presence: LabelPresence = LabelPresence.PRESENT
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError("class_name must not be blank")
        if "\n" in self.class_name or "\r" in self.class_name:
            raise ValueError(f"class_name must not contain line breaks. got {self.class_name!r}")
        if self.confidence is not None and not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1]. got {self.confidence}")

    @property
    def is_present(self) -> bool:
        """
        Whether the class is shown.

        Returns
        -------
        bool
            True if ``presence`` is ``PRESENT``.
        """
        return self.presence is LabelPresence.PRESENT
