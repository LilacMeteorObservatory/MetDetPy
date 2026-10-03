"""Read ordered image manifests shared by the photo CLI and development tools."""
import os
import warnings
from pathlib import Path

from .fileio import SUPPORT_ALL_IMG_FORMAT, is_ext_within


def load_image_manifest(filename: str) -> list[str]:
    """Read UTF-8 lines; relative paths are based on the manifest directory.

    Blank lines are ignored. Paths use the host platform's native syntax.
    Duplicate paths are skipped with a warning, preserving their first occurrence.
    Reject missing files and unsupported formats before inference.
    """
    manifest = Path(filename).resolve()
    images = []
    seen = set()
    for line_number, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        image = Path(line)
        if not image.is_absolute():
            image = manifest.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"{manifest}:{line_number}: {image}")
        if not is_ext_within(str(image), SUPPORT_ALL_IMG_FORMAT):
            raise ValueError(f"{manifest}:{line_number}: unsupported image: {image}")
        key = os.path.normcase(str(image))
        if key in seen:
            warnings.warn(f"{manifest}:{line_number}: duplicate image skipped: {image}",
                          UserWarning, stacklevel=2)
            continue
        seen.add(key)
        images.append(str(image))
    if not images:
        raise ValueError(f"Empty image manifest: {manifest}")
    return images
