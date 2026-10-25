from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ..annotation import AnnotationDataset, LabelCardinality


class AnnotationWriter(ABC):
    """
    Serializes an annotation dataset into one classification format.
    """

    @abstractmethod
    def write(self, dataset: AnnotationDataset, output_directory: Path) -> tuple[Path, ...]:
        """
        Write the dataset below ``output_directory``, creating the directory if needed.

        Parameters
        ----------
        dataset : AnnotationDataset
            Images, labels and class list to export.
        output_directory : Path
            Directory receiving the files.

        Returns
        -------
        tuple[Path, ...]
            Written files.
        """

    @staticmethod
    def _ensure_unique_paths(paths: Iterable[Path], description: str) -> None:
        path_counts: Counter[str] = Counter(path.as_posix().casefold() for path in paths)
        colliding_paths: list[str] = [path for path, count in path_counts.items() if count > 1]
        if colliding_paths:
            raise ValueError(f"{description} would collide, ignoring case: {colliding_paths}")

    @staticmethod
    def _ensure_file_name_safe(names: Iterable[str], description: str) -> None:
        name_tuple: tuple[str, ...] = tuple(names)
        unsafe_names: list[str] = [
            name for name in name_tuple if name in {".", ".."} or any(character in name for character in "/\\\0")
        ]
        if unsafe_names:
            raise ValueError(f"{description} cannot be used as file names: {unsafe_names}")
        AnnotationWriter._ensure_unique_paths((Path(name) for name in name_tuple), description)

    @staticmethod
    def _ensure_images_exist(dataset: AnnotationDataset) -> None:
        missing_paths: list[str] = [str(image.image_path) for image in dataset.images if not image.image_path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"image files not found: {missing_paths}")

    @staticmethod
    def _ensure_sources_preserved(dataset: AnnotationDataset, destinations: Iterable[Path]) -> None:
        source_paths: set[Path] = {image.image_path.resolve() for image in dataset.images}
        replacing_paths: list[str] = [
            str(destination)
            for destination in destinations
            if destination.parent.resolve() / destination.name in source_paths
        ]
        if replacing_paths:
            raise ValueError(f"destinations would replace source images: {replacing_paths}")

    @staticmethod
    def _ensure_single_label(dataset: AnnotationDataset, format_name: str) -> None:
        if dataset.label_cardinality is not LabelCardinality.SINGLE:
            raise ValueError(f"{format_name} needs a single-label dataset. got {dataset.label_cardinality}")

    @staticmethod
    def _relative_image_path(image_path: Path, directory: Path) -> str:
        return image_path.resolve().relative_to(directory.resolve(), walk_up=True).as_posix()
