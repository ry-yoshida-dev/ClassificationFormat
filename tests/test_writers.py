import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from classification_format import (
    AnnotationDataset,
    AnnotationFormat,
    CsvLayout,
    CsvWriter,
    HuggingFaceWriter,
    ImageAnnotation,
    ImageFolderWriter,
    ImageTransfer,
    LabelAnnotation,
    OpenImagesWriter,
    PascalVocWriter,
)
from classification_format.writer import JsonValue


def read_csv(path: Path) -> list[list[str]]:
    """
    Read a CSV file.

    Parameters
    ----------
    path : Path
        CSV file.

    Returns
    -------
    list[list[str]]
        Every row.
    """
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.reader(file))


def test_image_folder_places_images_in_class_directories(
    single_label_dataset: AnnotationDataset, tmp_path: Path
) -> None:
    output_directory: Path = tmp_path / "folder"
    written: tuple[Path, ...] = ImageFolderWriter().write(single_label_dataset, output_directory)
    assert written == (output_directory / "cat" / "cat.jpg", output_directory / "dog" / "dog.jpg")
    assert written[0].read_bytes() == b"cat.jpg"
    assert (output_directory / "bird").is_dir()


def test_image_folder_can_symlink(single_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    written: tuple[Path, ...] = ImageFolderWriter(ImageTransfer.SYMLINK).write(single_label_dataset, tmp_path / "f")
    assert written[0].is_symlink()
    assert written[0].resolve() == single_label_dataset.images[0].image_path.resolve()


def test_image_folder_can_rewrite_symlinks(single_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    writer: ImageFolderWriter = ImageFolderWriter(ImageTransfer.SYMLINK)
    writer.write(single_label_dataset, tmp_path / "f")
    written: tuple[Path, ...] = writer.write(single_label_dataset, tmp_path / "f")
    assert written[0].is_symlink()
    assert written[0].resolve() == single_label_dataset.images[0].image_path.resolve()


def test_image_folder_rejects_multi_label_dataset(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single-label"):
        ImageFolderWriter().write(multi_label_dataset, tmp_path / "folder")


def test_image_folder_rejects_class_names_unusable_as_directories(
    single_label_dataset: AnnotationDataset, tmp_path: Path
) -> None:
    unsafe: ImageAnnotation = replace(single_label_dataset.images[1], labels=(LabelAnnotation(class_name="a/b"),))
    unsafe_dataset: AnnotationDataset = replace(
        single_label_dataset, images=(single_label_dataset.images[0], unsafe), class_names=("cat", "dog", "a/b")
    )
    with pytest.raises(ValueError, match="file names"):
        ImageFolderWriter().write(unsafe_dataset, tmp_path / "folder")


def test_image_folder_refuses_missing_images(single_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    single_label_dataset.images[1].image_path.unlink()
    with pytest.raises(FileNotFoundError):
        ImageFolderWriter().write(single_label_dataset, tmp_path / "folder")
    assert not (tmp_path / "folder").exists()


def test_image_transfer_refuses_to_replace_its_source(single_label_dataset: AnnotationDataset) -> None:
    image_path: Path = single_label_dataset.images[0].image_path
    with pytest.raises(ValueError, match="source image itself"):
        ImageTransfer.COPY.apply(image_path, image_path)
    assert image_path.is_file()


def test_hugging_face_refuses_to_replace_sources_before_writing(multi_label_dataset: AnnotationDataset) -> None:
    image_directory: Path = multi_label_dataset.images[0].image_path.parent
    with pytest.raises(ValueError, match="replace source images"):
        HuggingFaceWriter().write(multi_label_dataset, image_directory)
    assert not (image_directory / "metadata.jsonl").exists()
    assert all(not image.image_path.is_symlink() and image.image_path.is_file() for image in multi_label_dataset.images)


def test_csv_binary_layout_marks_presence(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    written: tuple[Path, ...] = CsvWriter().write(multi_label_dataset, tmp_path / "csv")
    assert read_csv(written[0]) == [
        ["image_path", "person", "car", "dog"],
        ["../images/street.jpg", "1", "0", "-1"],
        ["../images/park.png", "1", "0", "1"],
        ["../images/empty.png", "0", "0", "0"],
    ]


def test_csv_binary_layout_can_leave_unlisted_classes_empty(
    multi_label_dataset: AnnotationDataset, tmp_path: Path
) -> None:
    rows: list[list[str]] = CsvWriter(is_unlisted_absent=False).build_rows(multi_label_dataset, tmp_path)
    assert rows[2] == ["images/park.png", "1", "", "1"]


def test_csv_label_list_layout_joins_present_classes(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    rows: list[list[str]] = CsvWriter(CsvLayout.LABEL_LIST).build_rows(multi_label_dataset, tmp_path)
    assert rows == [
        ["image_path", "labels"],
        ["images/street.jpg", "person"],
        ["images/park.png", "person|dog"],
        ["images/empty.png", ""],
    ]


def test_csv_label_list_rejects_class_names_containing_separator(
    multi_label_dataset: AnnotationDataset, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="separator"):
        CsvWriter(CsvLayout.LABEL_LIST, label_separator="o").build_rows(multi_label_dataset, tmp_path)


@pytest.mark.parametrize("file_name", ["", "..", "nested/labels.csv"])
def test_table_writers_reject_unusable_file_names(file_name: str) -> None:
    with pytest.raises(ValueError, match="file"):
        CsvWriter(file_name=file_name)
    with pytest.raises(ValueError, match="file"):
        OpenImagesWriter(file_name=file_name)


def test_csv_score_layout_writes_confidences(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    rows: list[list[str]] = CsvWriter(CsvLayout.SCORE, decimal_places=2).build_rows(multi_label_dataset, tmp_path)
    assert rows[1] == ["images/street.jpg", "0.90", "0.25", ""]


def test_hugging_face_writes_multi_label_metadata(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    output_directory: Path = tmp_path / "hf"
    written: tuple[Path, ...] = HuggingFaceWriter(is_confidence_included=True).write(
        multi_label_dataset, output_directory
    )
    assert [path.name for path in written] == ["street.jpg", "park.png", "empty.png", "metadata.jsonl"]
    assert (output_directory / "park.png").read_bytes() == b"park.png"
    records: list[JsonValue] = [json.loads(line) for line in written[-1].read_text(encoding="utf-8").splitlines()]
    assert records[0] == {"file_name": "street.jpg", "labels": ["person"], "scores": [0.9, 0.25, None]}
    assert records[2] == {"file_name": "empty.png", "labels": [], "scores": [None, None, None]}


def test_hugging_face_writes_single_label_metadata(single_label_dataset: AnnotationDataset) -> None:
    record: dict[str, JsonValue] = HuggingFaceWriter().build_record(
        single_label_dataset.images[0], single_label_dataset
    )
    assert record == {"file_name": "cat.jpg", "label": "cat"}


def test_hugging_face_rejects_colliding_file_names(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = multi_label_dataset.images[0]
    twin_path: Path = tmp_path / "other" / "STREET.jpg"
    twin_path.parent.mkdir()
    twin_path.write_bytes(b"twin")
    colliding: AnnotationDataset = replace(multi_label_dataset, images=(street, replace(street, image_path=twin_path)))
    with pytest.raises(ValueError, match="collide"):
        HuggingFaceWriter().write(colliding, tmp_path / "hf")


def test_pascal_voc_writes_image_sets(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    written: tuple[Path, ...] = PascalVocWriter(split_name="train").write(multi_label_dataset, tmp_path / "voc")
    image_set_directory: Path = tmp_path / "voc" / "ImageSets" / "Main"
    assert written == (
        image_set_directory / "train.txt",
        image_set_directory / "person_train.txt",
        image_set_directory / "car_train.txt",
        image_set_directory / "dog_train.txt",
    )
    assert written[0].read_text() == "street\npark\nempty\n"
    assert written[3].read_text() == "street  0\npark  1\nempty -1\n"


def test_pascal_voc_rejects_colliding_stems(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = multi_label_dataset.images[0]
    twin: ImageAnnotation = replace(street, image_path=street.image_path.with_name("Street.png"))
    with pytest.raises(ValueError, match="collide"):
        PascalVocWriter().write(replace(multi_label_dataset, images=(street, twin)), tmp_path / "voc")
    assert not (tmp_path / "voc").exists()


def test_pascal_voc_rejects_stems_containing_whitespace(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    street: ImageAnnotation = multi_label_dataset.images[0]
    spaced: ImageAnnotation = replace(street, image_path=street.image_path.with_name("my street.jpg"))
    with pytest.raises(ValueError, match="whitespace"):
        PascalVocWriter().write(replace(multi_label_dataset, images=(spaced,)), tmp_path / "voc")
    assert not (tmp_path / "voc").exists()


def test_open_images_separates_human_and_machine_labels(multi_label_dataset: AnnotationDataset, tmp_path: Path) -> None:
    written: tuple[Path, ...] = OpenImagesWriter(decimal_places=2).write(multi_label_dataset, tmp_path / "oi")
    assert read_csv(written[0]) == [
        ["ImageID", "Source", "LabelName", "Confidence"],
        ["street", "machine", "person", "0.90"],
        ["street", "machine", "car", "0.25"],
        ["park", "verification", "dog", "1"],
        ["park", "verification", "person", "1"],
    ]


@pytest.mark.parametrize("annotation_format", list(AnnotationFormat))
def test_every_format_writes_single_label_files(
    annotation_format: AnnotationFormat, single_label_dataset: AnnotationDataset, tmp_path: Path
) -> None:
    written: tuple[Path, ...] = annotation_format.create_writer().write(
        single_label_dataset, tmp_path / annotation_format.value
    )
    assert written
    assert all(path.is_file() for path in written)
    assert annotation_format.display_name


@pytest.mark.parametrize("annotation_format", [item for item in AnnotationFormat if item.is_multi_label_supported])
def test_multi_label_formats_write_files(
    annotation_format: AnnotationFormat, multi_label_dataset: AnnotationDataset, tmp_path: Path
) -> None:
    written: tuple[Path, ...] = annotation_format.create_writer().write(
        multi_label_dataset, tmp_path / annotation_format.value
    )
    assert all(path.is_file() for path in written)
