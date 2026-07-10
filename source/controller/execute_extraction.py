from collections.abc import Iterable, Iterator
from pathlib import Path

from source.controller.extract_field import ExtractField
from source.prompts.prompt_loader import load_prompt
from source.services.ai_client import AiClient


COURSE_FIELD_NAMES = ["curso"]
NAME_FIELD_NAMES = ["nome"]
NAME_FIELD_NAME = NAME_FIELD_NAMES
SIGNATURES_FIELD_NAMES = ["assinaturas"]
INSTITUTION_FIELD_NAME = ["instituicao"]

DEFAULT_PROMPT_VARIANT = "detailed_rules"


def execute_extraction(
        client: AiClient,
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
        client: AiClient,
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
        client: AiClient,
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
        client: AiClient,
        image_paths: Iterable[str | Path],
        model: str,
        prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> Iterator[dict]:
    return execute_extraction(
        client=client,
        image_paths=image_paths,
        model=model,
        task="assinaturas",
        field_names=SIGNATURES_FIELD_NAMES,
        prompt_variant=prompt_variant,
    )


def execute_university_extraction(
        client: AiClient,
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
