import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bson.json_util import dumps

from dotenv import load_dotenv
from source.controller.execute_extraction import (
    COURSE_FIELD_NAMES,
    INSTITUTION_FIELD_NAME,
    NAME_FIELD_NAMES,
    SIGNATURES_FIELD_NAMES,
    execute_course_extraction,
    execute_name_extraction,
    execute_signatures_extraction,
    execute_university_extraction,
    execute_nivel_extraction,
    NIVEL_NAME,
)
from source.models.mongo_client import DbConnectionHandler
from source.models.people_repository import PeopleRepository
from source.services.ai_client import AIClient
from source.services.evaluation_service import (
    CourseExtractionEvaluator,
    load_answer_key,
)
from source.services.image_service import iter_image_paths
from source.utils.console_formatter import (
    format_extraction_log,
    format_result,
    format_summary,
)

load_dotenv(Path(__file__).parent / "source" / ".env")

DATASET_PATH: str = str(
    Path(__file__).parent / "source" / "dataset-images" / "recortes_assinaturas_erros"
)
ANSWER_KEY_PATH: str = str(Path(__file__).parent / "source" / "answer_key.js")

CLIENT = AIClient(
    base_url=os.getenv("OPENAI_API_BASE", "http://home2.scanuto.com:8081"),
    email=os.getenv("OPENAI_API_USERNAME"),
    password=os.getenv("OPENAI_API_PASSWORD"),
    chat_completions_path=os.getenv("OPENAI_CHAT_PATH", "/ollama/api/chat"),
    timeout=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "300")),
)

MODELS: list[str] = [
    "qwen3.6:35b",
    "qwen3-vl:30b",
    "mistral-small3.2:latest",
    "gemma4:31b",
    "glm-ocr:latest",
    "gemma3:12b",
]


def main() -> dict[str, list[dict]]:
    # course_results = extract_course()
    # print("\n", flush=True)

    # name_results = extract_name()
    # print("\n", flush=True)

    signature_results, results = extract_signature()
    print("\n", flush=True)

    # print(results)

    # institution_results = extract_institution()
    # print("\n", flush=True)

    # curse_results = extract_course()
    # print("\n", flush=True)

    # nivel: list[dict[Any, Any]] = extract_nivel()

    return {
        "signature_results": signature_results,
    }


def extract_course():
    results = collect_results(
        execute_course_extraction(
            client=CLIENT,
            image_paths=iter_image_paths(DATASET_PATH),
            model=MODELS[2],
        ),
        label="curso",
    )

    return evaluate_results(results, COURSE_FIELD_NAMES)


def extract_name():
    results = collect_results(
        execute_name_extraction(
            client=CLIENT,
            image_paths=iter_image_paths(DATASET_PATH),
            model=MODELS[1],
        ),
        label="nome",
    )

    return evaluate_results(results, NAME_FIELD_NAMES)


def extract_signature():
    results = collect_results(
        execute_signatures_extraction(
            client=CLIENT,
            image_paths=iter_image_paths(DATASET_PATH, shuffle=True),
            model="qwen3-vl:30b",
            prompt_variant="zero_shot",
        ),
        label="as signature",
    )

    return evaluate_results(results, SIGNATURES_FIELD_NAMES), results


def extract_institution():
    results = collect_results(
        execute_university_extraction(
            client=CLIENT,
            image_paths=iter_image_paths(DATASET_PATH),
            model="gemma4:31b",
        ),
        label="institution",
    )

    return evaluate_results(results, INSTITUTION_FIELD_NAME)


def extract_nivel():
    results = collect_results(
        execute_nivel_extraction(
            client=CLIENT, image_paths=iter_image_paths(DATASET_PATH), model=MODELS[4]
        ),
        label="novel",
    )

    return evaluate_results(results, NIVEL_NAME)


def collect_results(results: Iterable[dict], label: str) -> list[dict]:
    collected = []

    for index, result in enumerate(results, start=1):
        file_name = (
            result.get("document_name") or Path(result.get("image_path", "")).name
        )
        raw_response = str(result.get("raw_response") or "")
        model = result.get("model")
        fields = result.get("fields", {})
        error = result.get("error")

        collected_result = {
            "documento": file_name,
            "document_name": file_name,
            "image_path": result.get("image_path"),
            "model": model,
            "task": result.get("task"),
            "prompt_variant": result.get("prompt_variant"),
            "raw_response": raw_response,
            "fields": fields,
            "error": error,
            "label": label,
            "campos": fields,
            "erro": error,
            "resposta_bruta": raw_response,
        }
        collected.append(collected_result)

        print(
            format_extraction_log(
                collected_result,
                index=index,
                label=label,
            ),
            flush=True,
        )

    return collected


def evaluate_results(results: list[dict], field_names: Iterable[str]) -> list[dict]:
    answer_key = load_answer_key(ANSWER_KEY_PATH)
    evaluator = CourseExtractionEvaluator(answer_key, field_names=field_names)
    evaluated_results = list(evaluator.evaluate_results(results))

    for resultado in evaluated_results:
        print(format_result(resultado))

    print(format_summary(evaluator.summary()))
    return evaluated_results


if __name__ == "__main__":
    main()
