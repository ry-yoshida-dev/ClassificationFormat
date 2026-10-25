from pathlib import Path

from ...annotation import AnnotationDataset, LabelAnnotation, LabelPresence
from ..base import AnnotationWriter


class PascalVocWriter(AnnotationWriter):
    """
    Writes Pascal VOC classification image sets below ``ImageSets/Main``.

    ``<split>.txt`` lists every image stem. ``<class>_<split>.txt`` has one line per image, the stem followed by
    ``1`` (present), ``-1`` (absent or not annotated) or ``0`` (ambiguous, VOC ``difficult``). Confidences are
    not kept.

    Parameters
    ----------
    split_name : str, optional
        Name of the image set, e.g. ``train``, ``val`` or ``trainval``.

    Raises
    ------
    ValueError
        If the split name is empty or not a valid file name.
    """

    image_set_directory: Path = Path("ImageSets") / "Main"

    def __init__(self, split_name: str = "trainval") -> None:
        if not split_name:
            raise ValueError("split_name must not be empty")
        self._ensure_file_name_safe((split_name,), "split name")
        self.split_name: str = split_name

    def build_lines(self, class_name: str, dataset: AnnotationDataset) -> list[str]:
        """
        Render the image set lines of one class.

        Parameters
        ----------
        class_name : str
            Listed class.
        dataset : AnnotationDataset
            Dataset to convert.

        Returns
        -------
        list[str]
            One ``"<stem> <value>"`` line per image, the value right-aligned in two characters.
        """
        return [
            f"{image.image_path.stem} {self._presence_value(image.find_label(class_name)):2d}"
            for image in dataset.images
        ]

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write the image set files.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            VOC root receiving ``ImageSets/Main``.

        Returns
        -------
        tuple[Path, ...]
            ``<split>.txt`` followed by one file per class in ``class_names`` order.

        Raises
        ------
        ValueError
            If an image stem contains whitespace, two images share a stem or a class name is not a valid file name,
            ignoring case.
        """
        self._ensure_file_name_safe(dataset.class_names, "class names")
        whitespace_stems: list[str] = [
            image.image_path.stem
            for image in dataset.images
            if any(character.isspace() for character in image.image_path.stem)
        ]
        if whitespace_stems:
            raise ValueError(f"image stems must not contain whitespace: {whitespace_stems}")
        self._ensure_unique_paths((Path(image.image_path.stem) for image in dataset.images), "image stems")
        image_set_directory: Path = output_directory / self.image_set_directory
        image_set_directory.mkdir(parents=True, exist_ok=True)
        split_path: Path = image_set_directory / f"{self.split_name}.txt"
        self._write_lines([image.image_path.stem for image in dataset.images], split_path)
        class_paths: list[Path] = []
        for class_name in dataset.class_names:
            class_path: Path = image_set_directory / f"{class_name}_{self.split_name}.txt"
            self._write_lines(self.build_lines(class_name, dataset), class_path)
            class_paths.append(class_path)
        return (split_path, *class_paths)

    @staticmethod
    def _presence_value(label: LabelAnnotation | None) -> int:
        if label is None:
            return -1
        match label.presence:
            case LabelPresence.PRESENT:
                return 1
            case LabelPresence.ABSENT:
                return -1
            case LabelPresence.AMBIGUOUS:
                return 0

    @staticmethod
    def _write_lines(lines: list[str], output_path: Path) -> None:
        output_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")
