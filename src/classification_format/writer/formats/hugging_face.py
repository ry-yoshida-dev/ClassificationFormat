import json
from pathlib import Path

from ...annotation import AnnotationDataset, ImageAnnotation, LabelAnnotation, LabelCardinality
from ..base import AnnotationWriter
from ..image_transfer import ImageTransfer
from ..json_value import JsonValue


class HuggingFaceWriter(AnnotationWriter):
    """
    Writes a Hugging Face ``imagefolder`` dataset with a ``metadata.jsonl`` file.

    Images are placed next to the metadata file, keeping their file names, so that ``file_name`` resolves.
    Single-label records carry ``label`` (a class name); multi-label records carry ``labels`` (the present classes).

    Parameters
    ----------
    image_transfer : ImageTransfer, optional
        Whether images are copied or linked.
    is_confidence_included : bool, optional
        Whether to add ``scores``, the confidence of every class in ``class_names`` order, ``null`` where missing.
    """

    metadata_file_name: str = "metadata.jsonl"

    def __init__(
        self, image_transfer: ImageTransfer = ImageTransfer.COPY, is_confidence_included: bool = False
    ) -> None:
        self.image_transfer: ImageTransfer = image_transfer
        self.is_confidence_included: bool = is_confidence_included

    def build_record(self, image: ImageAnnotation, dataset: AnnotationDataset) -> dict[str, JsonValue]:
        """
        Build the metadata record of one image.

        Parameters
        ----------
        image : ImageAnnotation
            Image to render.
        dataset : AnnotationDataset
            Dataset providing the cardinality and class order.

        Returns
        -------
        dict[str, JsonValue]
            Record with ``file_name`` and the labels.
        """
        record: dict[str, JsonValue] = {"file_name": image.image_path.name}
        match dataset.label_cardinality:
            case LabelCardinality.SINGLE:
                record["label"] = image.present_class_names[0]
            case LabelCardinality.MULTI:
                record["labels"] = list(image.present_class_names)
        if self.is_confidence_included:
            scores: list[JsonValue] = []
            for class_name in dataset.class_names:
                label: LabelAnnotation | None = image.find_label(class_name)
                scores.append(None if label is None else label.confidence)
            record["scores"] = scores
        return record

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Place the images and write the metadata file.

        Parameters
        ----------
        dataset : AnnotationDataset
            Dataset to export.
        output_directory : Path
            Directory receiving the images and the metadata file.

        Returns
        -------
        tuple[Path, ...]
            Placed images in image order followed by ``metadata.jsonl``.

        Raises
        ------
        ValueError
            If two images share a file name or an image is named like the metadata file, ignoring case, or a placed
            image would replace a source image.
        FileNotFoundError
            If an image file does not exist.
        """
        self._ensure_images_exist(dataset)
        metadata_path: Path = output_directory / self.metadata_file_name
        destinations: tuple[Path, ...] = tuple(output_directory / image.image_path.name for image in dataset.images)
        self._ensure_unique_paths((*destinations, metadata_path), "image and metadata files")
        self._ensure_sources_preserved(dataset, (*destinations, metadata_path))
        output_directory.mkdir(parents=True, exist_ok=True)
        for image, destination in zip(dataset.images, destinations, strict=True):
            self.image_transfer.apply(image.image_path, destination)
        lines: list[str] = [
            json.dumps(self.build_record(image, dataset), ensure_ascii=False) + "\n" for image in dataset.images
        ]
        metadata_path.write_text("".join(lines), encoding="utf-8", newline="\n")
        return (*destinations, metadata_path)
