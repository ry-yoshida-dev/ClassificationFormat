import csv
from pathlib import Path

from ...annotation import AnnotationDataset, LabelAnnotation, LabelPresence
from ..base import AnnotationWriter


class OpenImagesWriter(AnnotationWriter):
    """
    Writes Open Images image-level labels as one CSV file.

    Columns are ``ImageID,Source,LabelName,Confidence``; ``ImageID`` is the image stem and ``LabelName`` the class
    name. Labels with a confidence become machine-generated rows holding the score, whether present or absent, so
    the presence decided for scored labels (e.g. by ``ImageAnnotation.from_scores``) is not kept; readers threshold
    the score themselves. Labels without a confidence become human-verified rows holding ``1`` (present) or ``0``
    (absent). Ambiguous labels and classes not annotated in an image are omitted, which Open Images reads as
    unverified.

    Parameters
    ----------
    file_name : str, optional
        Name of the CSV file.
    human_source : str, optional
        ``Source`` of labels without a confidence.
    machine_source : str, optional
        ``Source`` of labels with a confidence.
    decimal_places : int, optional
        Digits after the decimal point of machine confidences.

    Raises
    ------
    ValueError
        If the file name is empty or not a valid file name, or ``decimal_places`` is negative.
    """

    header: tuple[str, ...] = ("ImageID", "Source", "LabelName", "Confidence")

    def __init__(
        self,
        file_name: str = "image_labels.csv",
        human_source: str = "verification",
        machine_source: str = "machine",
        decimal_places: int = 6,
    ) -> None:
        if not file_name:
            raise ValueError("file_name must not be empty")
        self._ensure_file_name_safe((file_name,), "file name")
        if decimal_places < 0:
            raise ValueError(f"decimal_places must not be negative. got {decimal_places}")
        self.file_name: str = file_name
        self.human_source: str = human_source
        self.machine_source: str = machine_source
        self.decimal_places: int = decimal_places

    def build_rows(self, dataset: AnnotationDataset) -> list[list[str]]:
        """
        Render the table.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to convert.

        Returns
        -------
        list[list[str]]
            Header row followed by one row per present or absent label, in image and label order.
        """
        rows: list[list[str]] = [list(self.header)]
        for image in dataset.images:
            for label in image.labels:
                match label.presence:
                    case LabelPresence.AMBIGUOUS:
                        continue
                    case LabelPresence.PRESENT | LabelPresence.ABSENT:
                        rows.append([image.image_path.stem, *self._source_and_confidence(label)])
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
            If two images share a stem, ignoring case.
        """
        self._ensure_unique_paths((Path(image.image_path.stem) for image in dataset.images), "image stems")
        rows: list[list[str]] = self.build_rows(dataset)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path: Path = output_directory / self.file_name
        with output_path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file, lineterminator="\n").writerows(rows)
        return (output_path,)

    def _source_and_confidence(self, label: LabelAnnotation) -> list[str]:
        if label.confidence is not None:
            return [self.machine_source, label.class_name, f"{label.confidence:.{self.decimal_places}f}"]
        return [self.human_source, label.class_name, "1" if label.is_present else "0"]
