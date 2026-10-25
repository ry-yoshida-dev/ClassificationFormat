from enum import Enum


class CsvLayout(Enum):
    """
    Column layout of a classification CSV table.

    ``LABEL_LIST`` writes one ``labels`` column joining the present classes in ``class_names`` order. ``BINARY``
    writes one column per class holding ``1`` (present), ``0`` (absent) or ``-1`` (ambiguous), as in CheXpert.
    ``SCORE`` writes one column per class holding the confidence.
    """

    LABEL_LIST = "label_list"
    BINARY = "binary"
    SCORE = "score"
