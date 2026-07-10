import base64
import mimetypes
import re
from pathlib import Path
from typing import Iterator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def image_to_base64(image_path: str | Path ) -> str:
    image_path = Path(image_path)

    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpg"


    image_base64 = image_to_base64(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def iter_image_paths(dataset_path: str | Path) -> Iterator[Path]:
    dataset_path = Path(dataset_path)

    for path in sorted(dataset_path.iterdir(), key=_natural_name_key):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _natural_name_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]
