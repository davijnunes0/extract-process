import argparse
import json
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote

from source.services.image_service import iter_image_paths
from source.services.tesseract_signature_service import (
    SignatureLocalizationError,
    crop_fallback_signature_region,
    execute_ocr_on_signature_region,
    load_and_validate_image,
    locate_and_crop_signature_region,
    save_processing_image,
    serialize_localization,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "source" / "dataset-images" / "dataset_completo"
DATASET_CATEGORIES = (
    "imagens_com_assinatura",
    "imagens_ortogonais",
    "imagens_sem_assinatura_categoria",
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recorta documentos por categoria usando somente imagens originais."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output", type=Path, default=None, help="Padrão: <dataset>/tesseract."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Filtro opcional de prefixos. Por padrão, processa todos os nomes.",
    )
    parser.add_argument("--language", default="por")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args(arguments)


def _build_compact_result(record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "image": record["image"],
        "category": record["category"],
        "status": record["status"],
        "original_path": record.get("original_path"),
        "crop": {
            "path": record.get("crop_path"),
            "method": record.get("crop_method"),
            "pixels": record.get("box", []),
            "normalized_1000": record.get("coordenadas_recorte", []),
        },
        "text": record.get("ocr_text", ""),
    }
    for key in ("error", "crop_ocr_error"):
        if record.get(key):
            payload[key] = record[key]
    return payload


def collect_crop(
    crop_path: Path,
    output_root: Path,
    category: str,
    relative_path: Path,
) -> Path:
    """Copia qualquer recorte para a pasta única, incluindo os de fallback.

    Categoria, caminho relativo e extensão identificam o documento. Separadores
    de subpastas são codificados para manter todos os PNGs no mesmo diretório.
    A cópia por documento permanece intacta. A pasta inclui exemplos positivos
    e negativos; o OCR não decide se existe assinatura manuscrita.
    """
    filename = (
        f"{quote(category, safe='')}__{quote(relative_path.as_posix(), safe='')}.png"
    )
    destination = output_root / "recortes_localizados" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(crop_path, destination)
    return destination


def process_image(
    image_path: Path,
    output_root: Path,
    language: str,
    category: str = "imagens_com_assinatura",
    relative_path: Path | None = None,
) -> dict[str, Any]:
    """Copia o arquivo original e salva recorte e JSON sem pré-processamento.

    A categoria é herdada da pasta de entrada, não inferida pelo OCR.
    Ausência de palavra-chave não comprova ausência de assinatura.
    """
    relative_path = relative_path or Path(image_path.name)
    # A extensão no nome da pasta evita colisões entre IQ_1.jpg e IQ_1.png.
    destination = output_root / category / relative_path
    record: dict[str, Any] = {
        "image": relative_path.as_posix(),
        "category": category,
        "group": image_path.stem.split("_", 1)[0],
        "status": "error",
    }
    try:
        destination.mkdir(parents=True, exist_ok=True)
        original_path = destination / f"original{image_path.suffix}"
        shutil.copy2(image_path, original_path)
        record["original_path"] = str(original_path)
        image = load_and_validate_image(image_path)
        try:
            region = locate_and_crop_signature_region(image, language=language)
            record.update(status="localized", crop_method="anchor")
        except SignatureLocalizationError as error:
            region = crop_fallback_signature_region(image)
            record.update(
                status="not_localized", crop_method="fallback", error=str(error)
            )

        crop_path = save_processing_image(region.image, destination / "recorte.png")
        record["crop_path"] = str(crop_path)
        collect_crop(crop_path, output_root, category, relative_path)
        record.update(serialize_localization(region, image.width, image.height))
        try:
            ocr_result = execute_ocr_on_signature_region(
                region.image, language=language
            )
            record["ocr_text"] = ocr_result.text
        except SignatureLocalizationError as error:
            record["crop_ocr_error"] = str(error)
            record["status"] = "error"
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")

    if destination.is_dir():
        localization_path = destination / "localizacao.json"
        localization_path.write_text(
            json.dumps(_build_compact_result(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        record["localization_path"] = str(localization_path)
    return record


def write_group_report(
    group: str,
    records: Sequence[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """Salva um relatório compacto para a categoria de entrada."""
    ordered = sorted(records, key=lambda record: record["image"])
    summary = {
        "category": group,
        "images_total": len(ordered),
        "localized": sum(record["status"] == "localized" for record in ordered),
        "not_localized": sum(record["status"] == "not_localized" for record in ordered),
        "errors": sum(record["status"] == "error" for record in ordered),
    }
    destination = output_root / group
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "relatorio.json").write_text(
        json.dumps(
            {"summary": summary, "images": [_build_compact_result(r) for r in ordered]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def run_tesseract_pipeline(
    dataset_path: Path,
    output_path: Path,
    groups: Sequence[str] | None,
    language: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Processa as categorias configuradas sem alterar nem mover as entradas."""
    dataset_path = dataset_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset inexistente: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Não é um diretório: {dataset_path}")
    if output_path == dataset_path or output_path in dataset_path.parents:
        raise ValueError("A saída não pode ser o dataset nem um ancestral dele.")
    if workers <= 0:
        raise ValueError("workers deve ser maior que zero.")
    for category in DATASET_CATEGORIES:
        category_root = dataset_path / category
        if output_path == category_root or output_path in category_root.parents:
            raise ValueError("A saída não pode substituir uma categoria de entrada.")
    normalized_groups = (
        None
        if groups is None
        else {group.strip().upper() for group in groups if group.strip()}
    )
    if normalized_groups == set():
        raise ValueError("Informe pelo menos um grupo válido.")

    jobs = []
    for category in DATASET_CATEGORIES:
        category_root = dataset_path / category
        if not category_root.is_dir():
            continue
        for path in iter_image_paths(category_root):
            if path.resolve().is_relative_to(output_path):
                continue
            if normalized_groups is not None and (
                path.stem.split("_", 1)[0].upper() not in normalized_groups
            ):
                continue
            jobs.append((path, category, path.relative_to(category_root)))

    print(f"Dataset: {dataset_path}\nSaída: {output_path}\nImagens: {len(jobs)}")
    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_image, path, output_path, language, category, relative
            )
            for path, category, relative in jobs
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(jobs)}] {record['category']}/{record['image']}: "
                f"{record['status']}",
                flush=True,
            )
    for category in DATASET_CATEGORIES:
        summary = write_group_report(
            category, [r for r in records if r["category"] == category], output_path
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return records


def main(arguments: Sequence[str] | None = None) -> int:
    configuration = parse_arguments(arguments)
    output_path = configuration.output or configuration.dataset / "tesseract"
    try:
        records = run_tesseract_pipeline(
            dataset_path=configuration.dataset,
            output_path=output_path,
            groups=configuration.groups,
            language=configuration.language,
            workers=configuration.workers,
        )
    except (OSError, ValueError) as error:
        print(f"Erro de configuração: {error}")
        return 1
    return int(any(record["status"] == "error" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
