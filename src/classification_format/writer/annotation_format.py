from enum import Enum

from .base import AnnotationWriter
from .formats import CsvWriter, HuggingFaceWriter, ImageFolderWriter, OpenImagesWriter, PascalVocWriter


class AnnotationFormat(Enum):
    """
    Supported export formats.
    """

    IMAGE_FOLDER = "image_folder"
    CSV = "csv"
    HUGGING_FACE = "hugging_face"
    PASCAL_VOC = "pascal_voc"
    OPEN_IMAGES = "open_images"

    @property
    def display_name(self) -> str:
        """
        Human-readable format name.

        Returns
        -------
        str
            Name suitable for menus.
        """
        match self:
            case AnnotationFormat.IMAGE_FOLDER:
                return "ImageFolder (class directories)"
            case AnnotationFormat.CSV:
                return "CSV (one column per class)"
            case AnnotationFormat.HUGGING_FACE:
                return "Hugging Face imagefolder (JSONL)"
            case AnnotationFormat.PASCAL_VOC:
                return "Pascal VOC image sets (TXT)"
            case AnnotationFormat.OPEN_IMAGES:
                return "Open Images image labels (CSV)"

    @property
    def is_multi_label_supported(self) -> bool:
        """
        Whether the format can hold multi-label datasets.

        Returns
        -------
        bool
            False for formats storing exactly one class per image.
        """
        match self:
            case AnnotationFormat.IMAGE_FOLDER:
                return False
            case (
                AnnotationFormat.CSV
                | AnnotationFormat.HUGGING_FACE
                | AnnotationFormat.PASCAL_VOC
                | AnnotationFormat.OPEN_IMAGES
            ):
                return True

    def create_writer(self) -> AnnotationWriter:
        """
        Build a writer of this format with default options.

        Returns
        -------
        AnnotationWriter
            Writer producing this format.
        """
        match self:
            case AnnotationFormat.IMAGE_FOLDER:
                return ImageFolderWriter()
            case AnnotationFormat.CSV:
                return CsvWriter()
            case AnnotationFormat.HUGGING_FACE:
                return HuggingFaceWriter()
            case AnnotationFormat.PASCAL_VOC:
                return PascalVocWriter()
            case AnnotationFormat.OPEN_IMAGES:
                return OpenImagesWriter()
