import shutil
from enum import Enum
from pathlib import Path


class ImageTransfer(Enum):
    """
    How writers place image files inside the output directory.
    """

    COPY = "copy"
    SYMLINK = "symlink"
    HARDLINK = "hardlink"

    def apply(self, source: Path, destination: Path) -> None:
        """
        Place one image file, replacing an existing file at the destination.

        Parameters
        ----------
        source : Path
            Existing image file.
        destination : Path
            Location inside an existing directory; symbolic links point to the resolved source.

        Raises
        ------
        ValueError
            If the destination is the source file itself.
        """
        if destination.parent.resolve() / destination.name == source.resolve():
            raise ValueError(f"destination {destination} is the source image itself")
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        match self:
            case ImageTransfer.COPY:
                shutil.copy2(source, destination)
            case ImageTransfer.SYMLINK:
                destination.symlink_to(source.resolve())
            case ImageTransfer.HARDLINK:
                destination.hardlink_to(source)
