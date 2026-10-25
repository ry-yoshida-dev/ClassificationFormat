from pathlib import Path

from ...annotation import AnnotationDataset, ImageAnnotation
from ..base import AnnotationWriter
from ..image_transfer import ImageTransfer


class ImageFolderWriter(AnnotationWriter):
    """
    Places each image in a directory named after its class.

    This is the layout read by torchvision ``ImageFolder``, Keras ``image_dataset_from_directory``, Ultralytics
    classification and Create ML. A directory is created for every listed class, including unused ones. Those
    loaders number classes by sorted directory name, not by ``class_names`` order. Only single-label datasets fit
    this layout, and confidences are not kept.

    Parameters
    ----------
    image_transfer : ImageTransfer, optional
        Whether images are copied or linked.
    """

    format_name: str = "ImageFolder"

    def __init__(self, image_transfer: ImageTransfer = ImageTransfer.COPY) -> None:
        self.image_transfer: ImageTransfer = image_transfer

    def image_destination(self, image: ImageAnnotation, output_directory: Path) -> Path:
        """
        Location of one image inside the output directory.

        Parameters
        ----------
        image : ImageAnnotation
            Image of a single-label dataset.
        output_directory : Path
            Directory receiving the class directories.

        Returns
        -------
        Path
            ``output_directory / class name / image file name``.
        """
        return output_directory / image.present_class_names[0] / image.image_path.name

    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Place every image in its class directory.

        Parameters
        ----------
        dataset : AnnotationDataset
            Single-label dataset to export.
        output_directory : Path
            Directory receiving the class directories.

        Returns
        -------
        tuple[Path, ...]
            Placed images in image order.

        Raises
        ------
        ValueError
            If the dataset is multi-label, a class name is not a valid directory name, or two images of one class
            share a file name, ignoring case, or a placed image would replace a source image.
        FileNotFoundError
            If an image file does not exist.
        """
        self._ensure_single_label(dataset, self.format_name)
        self._ensure_file_name_safe(dataset.class_names, "class names")
        self._ensure_images_exist(dataset)
        destinations: tuple[Path, ...] = tuple(
            self.image_destination(image, output_directory) for image in dataset.images
        )
        self._ensure_unique_paths(destinations, "images of one class")
        self._ensure_sources_preserved(dataset, destinations)
        for class_name in dataset.class_names:
            (output_directory / class_name).mkdir(parents=True, exist_ok=True)
        for image, destination in zip(dataset.images, destinations, strict=True):
            self.image_transfer.apply(image.image_path, destination)
        return destinations
