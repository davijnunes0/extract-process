import base64
import mimetypes
import random
import re
from pathlib import Path
from typing import Iterator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def image_to_base64(image_path: str | Path) -> str:
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


def iter_image_paths(
    dataset_path: str | Path,
    shuffle: bool = False,
) -> Iterator[Path]:
    """Lista imagens do dataset em ordem natural, opcionalmente embaralhada.

    ``shuffle=True`` mistura positivos e negativos antes do envio ao modelo,
    evitando que a ordem agrupada por categoria enviese a extração.
    """
    dataset_path = Path(dataset_path)

    paths = []
    for path in sorted(dataset_path.rglob("*"), key=_natural_name_key):
        relative_path = path.relative_to(dataset_path)

        # Não processa novamente os resultados gerados pelo pipeline.
        if relative_path.parts and relative_path.parts[0] == "tesseract":
            continue

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    if shuffle:
        random.shuffle(paths)

    yield from paths


def _natural_name_key(path: Path) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]
