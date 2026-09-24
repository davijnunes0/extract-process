from collections.abc import Iterable, Iterator
from pathlib import Path

from source.controller.extract_field import ExtractField
from source.prompts.prompt_loader import load_prompt
from source.services.ai_client import AIClient


COURSE_FIELD_NAMES = ["curso"]
NAME_FIELD_NAMES = ["nome"]
SIGNATURES_FIELD_NAMES = ["assinaturas"]
SIGNATURE_EXTRACTION_FIELD_NAMES = [
    "assinaturas",
    "trecho_verificado",
    "coordenadas_recorte",
]
ZERO_SHOT_SIGNATURE_FIELD_NAMES = ["tem_assinatura"]
INSTITUTION_FIELD_NAME = ["instituicao"]
NIVEL_NAME = ["nivel"]

DEFAULT_PROMPT_VARIANT = "detailed_rules"
ZERO_SHOT_PROMPT_VARIANT = "zero_shot"


def execute_extraction(
    client: AIClient,
    image_paths: Iterable[str | Path],
    model: str,
    task: str,
    field_names: Iterable[str],
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> Iterator[dict]:
    prompt = load_prompt(task=task, variant=prompt_variant)

    for result in ExtractField.extract_images(
        client=client,
        image_paths=image_paths,
        model=model,
        prompt=prompt,
        field_names=field_names,
    ):
        result["task"] = task
        result["prompt_variant"] = prompt_variant
        yield result


def execute_course_extraction(
    client: AIClient,
    image_paths: Iterable[str | Path],
    model: str,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> Iterator[dict]:
    return execute_extraction(
        client=client,
        image_paths=image_paths,
        model=model,
        task="curso",
        field_names=COURSE_FIELD_NAMES,
        prompt_variant=prompt_variant,
    )


def execute_name_extraction(
    client: AIClient,
    image_paths: Iterable[str | Path],
    model: str,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> Iterator[dict]:
    return execute_extraction(
        client=client,
        image_paths=image_paths,
        model=model,
        task="nome",
        field_names=NAME_FIELD_NAMES,
        prompt_variant=prompt_variant,
    )


def execute_signatures_extraction(
    client: AIClient,
    image_paths: Iterable[str | Path],
    model: str,
    prompt_variant: str = ZERO_SHOT_PROMPT_VARIANT,
) -> Iterator[dict]:
    field_names = (
        ZERO_SHOT_SIGNATURE_FIELD_NAMES
        if prompt_variant == ZERO_SHOT_PROMPT_VARIANT
        else SIGNATURE_EXTRACTION_FIELD_NAMES
    )

    for result in execute_extraction(
        client=client,
        image_paths=image_paths,
        model=model,
        task="assinaturas",
        field_names=field_names,
        prompt_variant=prompt_variant,
    ):
        yield _with_signature_decision(result)


def _with_signature_decision(result: dict) -> dict:
    fields = dict(result.get("fields") or {})
    if fields.get("assinaturas") is None:
        fields["assinaturas"] = fields.get("tem_assinatura")
    result["fields"] = fields
    result["campos"] = fields
    return result


def execute_university_extraction(
    client: AIClient,
    image_paths: Iterable[str | Path],
    model: str,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> Iterator[dict]:
    return execute_extraction(
        client=client,
        image_paths=image_paths,
        model=model,
        task="instituicao",
        field_names=INSTITUTION_FIELD_NAME,
        prompt_variant=prompt_variant,
    )


def execute_nivel_extraction(
    client: AIClient,
    image_paths: Iterable[str | Path],
    model: str,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> Iterator[dict]:
    return execute_extraction(
        client=client,
        image_paths=image_paths,
        model=model,
        task="nivel",
        field_names=NIVEL_NAME,
        prompt_variant=prompt_variant,
    )
