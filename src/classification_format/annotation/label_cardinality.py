from enum import Enum


class LabelCardinality(Enum):
    """
    Number of classes an image may show.

    ``SINGLE`` requires exactly one present class per image, as softmax classifiers and class-per-directory formats
    assume. ``MULTI`` allows any number, including none, as sigmoid classifiers assume.
    """

    SINGLE = "single"
    MULTI = "multi"
