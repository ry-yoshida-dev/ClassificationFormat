import csv
from pathlib import Path

from ...annotation import AnnotationDataset, ImageAnnotation, LabelAnnotation, LabelPresence
from ..base import AnnotationWriter
from .csv_layout import CsvLayout


class CsvWriter(AnnotationWriter):
    """
    Writes one CSV table with a row per image.

    The first column holds the image path relative to the output directory. The remaining columns follow ``layout``;
    class columns are in ``class_names`` order. Empty cells mark classes without a value.

    Parameters
    ----------
    layout : CsvLayout, optional
        Columns describing the labels.
    file_name : str, optional
        Name of the CSV file.
    label_separator : str, optional
        Separator joining the present classes of ``LABEL_LIST`` in ``class_names`` order.
    is_unlisted_absent : bool, optional
        Whether ``BINARY`` writes classes not annotated in an image as ``0`` instead of an empty cell.
    decimal_places : int, optional
        Digits after the decimal point of ``SCORE`` values.

    Raises
    ------
    ValueError
        If the file name is empty or not a valid file name, the label separator is empty or contains a line break, or
        ``decimal_places`` is negative.
    """

    image_column_name: str = "image_path"
    labels_column_name: str = "labels"

    def __init__(
        self,
        layout: CsvLayout = CsvLayout.BINARY,
        file_name: str = "labels.csv",
        label_separator: str = "|",
        is_unlisted_absent: bool = True,
        decimal_places: int = 6,
    ) -> None:
        if not file_name:
            raise ValueError("file_name must not be empty")
        self._ensure_file_name_safe((file_name,), "file name")
        if not label_separator or "\n" in label_separator or "\r" in label_separator:
            raise ValueError(f"label_separator must be non-empty without line breaks. got {label_separator!r}")
        if decimal_places < 0:
            raise ValueError(f"decimal_places must not be negative. got {decimal_places}")
        self.layout: CsvLayout = layout
        self.file_name: str = file_name
        self.label_separator: str = label_separator
        self.is_unlisted_absent: bool = is_unlisted_absent
        self.decimal_places: int = decimal_places

    def build_rows(self, dataset: AnnotationDataset, output_directory: Path) -> list[list[str]]:
        """
        Render the table.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to convert.
        output_directory : Path
            Directory the CSV file is written to, used to relativize image paths.

        Returns
        -------
        list[list[str]]
            Header row followed by one row per image.

        Raises
        ------
        ValueError
            If a class name contains the label separator in ``LABEL_LIST``, or equals the image column name.
        """
        match self.layout:
            case CsvLayout.LABEL_LIST:
                separated_class_names: list[str] = [
                    class_name for class_name in dataset.class_names if self.label_separator in class_name
                ]
                if separated_class_names:
                    raise ValueError(
                        f"class names contain the label separator {self.label_separator!r}: {separated_class_names}"
                    )
                header: list[str] = [self.image_column_name, self.labels_column_name]
            case CsvLayout.BINARY | CsvLayout.SCORE:
                if self.image_column_name in dataset.class_names:
                    raise ValueError(f"class name {self.image_column_name!r} collides with the image column")
                header = [self.image_column_name, *dataset.class_names]
        rows: list[list[str]] = [header]
        for image in dataset.images:
            rows.append(
                [self._relative_image_path(image.image_path, output_directory), *self._build_cells(image, dataset)]
            )
        return rows

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write the CSV file as UTF-8.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            Directory receiving the file.

        Returns
        -------
        tuple[Path, ...]
            The CSV file.

        Raises
        ------
        ValueError
            If a class name contains the label separator in ``LABEL_LIST``, or equals the image column name.
        """
        rows: list[list[str]] = self.build_rows(dataset, output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path: Path = output_directory / self.file_name
        with output_path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file, lineterminator="\n").writerows(rows)
        return (output_path,)

    def _build_cells(self, image: ImageAnnotation, dataset: AnnotationDataset) -> list[str]:
        match self.layout:
            case CsvLayout.LABEL_LIST:
                present_class_names: set[str] = set(image.present_class_names)
                return [
                    self.label_separator.join(
                        class_name for class_name in dataset.class_names if class_name in present_class_names
                    )
                ]
            case CsvLayout.BINARY:
                return [self._binary_cell(image.find_label(class_name)) for class_name in dataset.class_names]
            case CsvLayout.SCORE:
                return [self._score_cell(image.find_label(class_name)) for class_name in dataset.class_names]

    def _binary_cell(self, label: LabelAnnotation | None) -> str:
        if label is None:
            return "0" if self.is_unlisted_absent else ""
        match label.presence:
            case LabelPresence.PRESENT:
                return "1"
            case LabelPresence.ABSENT:
                return "0"
            case LabelPresence.AMBIGUOUS:
                return "-1"

    def _score_cell(self, label: LabelAnnotation | None) -> str:
        if label is None or label.confidence is None:
            return ""
        return f"{label.confidence:.{self.decimal_places}f}"
