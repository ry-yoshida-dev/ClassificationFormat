from enum import Enum


class LabelPresence(Enum):
    """
    Annotated state of one class in one image.

    ``PRESENT`` and ``ABSENT`` are confirmed decisions. ``AMBIGUOUS`` marks a class that was looked at but could not
    be decided, such as Pascal VOC ``difficult`` (``0``) or an uncertain finding (``-1`` in CheXpert-style tables).
    A class without any label in an image is not annotated at all.
    """

    PRESENT = "present"
    ABSENT = "absent"
    AMBIGUOUS = "ambiguous"
