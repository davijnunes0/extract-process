import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from bson.json_util import dumps

from dotenv import load_dotenv
from source.controller.execute_extraction import (
    execute_course_extraction,
    execute_name_extraction,
    execute_signatures_extraction,
    execute_university_extraction
)
from source.models.mongo_client import DbConnectionHandler
from source.models.people_repository import PeopleRepository
from source.services.ai_client import AiClient
from source.services.evaluation_service import CourseExtractionEvaluator, load_answer_key
from source.services.image_service import iter_image_paths
from source.utils.console_formatter import format_result, format_summary

load_dotenv()

dataset_path: str = r"C:\Users\davijnunes\Documents\Python\course-extract\dataset-images"
answer_key_path: str = r"C:\Users\davijnunes\Documents\Python\course-extract\answer_key.js"

client = AiClient(
    base_url=os.getenv("OPENAI_API_BASE", "http://home2.scanuto.com:8081"),
    email=os.getenv("OPENAI_API_USERNAME", "davijnunesdeveloper@gmail.com"),
    password=os.getenv("OPENAI_API_PASSWORD", "Davinunes01$"),
    chat_completions_path=os.getenv("OPENAI_CHAT_PATH", "/ollama/api/chat"),
    timeout=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "1800")),
)


def main():
    extract_course()
    print("\n", flush=True)

    extract_name()
    print("\n", flush=True)

    extract_signature()
    print("\n", flush=True)

    extract_institution()
    print("\n", flush=True)

def extract_course():
    results = collect_results(
        execute_course_extraction(
            client=client,
            image_paths=iter_image_paths(dataset_path),
            model="gemma4:31b",
        ),
        label="curso",
    )

    return results


def extract_name():
    results = collect_results(
        execute_name_extraction(
            client=client,
            image_paths=iter_image_paths(dataset_path),
            model="gemma4:31b",
        ),
        label="nome",
    )

    return results


def extract_signature():
    results = collect_results(
        execute_signatures_extraction(
            client=client,
            image_paths=iter_image_paths(dataset_path),
            model="ministral-3:14b",
        ),
        label="assinatura"
    )

    return results


def extract_institution():
    results = collect_results(
        execute_university_extraction(
            client=client,
            image_paths=iter_image_paths(dataset_path),
            model="gemma4:31b",
        ),
        label="instituição",
    )

    return results


def collect_results(results: Iterable[dict], label: str) -> list[dict]:
    collected = []

    for index, result in enumerate(results, start=1):
        file_name = result.get("document_name") or Path(result.get("image_path", "")).name
        raw_response = str(result.get("raw_response") or "")
        model = result.get("model")
        fields = result.get("fields", {})
        error = result.get("error")

        collected_result = {
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

        print(f"[{label}] {index} {file_name}: {fields}", flush=True)
        print(f"[{label}] {index} model: {model}", flush=True)

        if error:
            print(f"[{label}] {index} erro: {error}", flush=True)

        print(
            f"[{label}] {index} resposta_bruta: {raw_response[:2000]!r}",
            flush=True,
        )

    return collected


def evaluate_results(results: list[dict], field_names: Iterable[str]) -> None:
    answer_key = load_answer_key(answer_key_path)
    evaluator = CourseExtractionEvaluator(answer_key, field_names=field_names)

    for resultado in evaluator.evaluate_results(results):
        print(format_result(resultado))

    print(format_summary(evaluator.summary()))


if __name__ == "__main__":
    main()
