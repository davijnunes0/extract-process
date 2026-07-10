import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path


DEFAULT_FIELD_NAMES = ["curso"]


class CourseExtractionEvaluator:
    def __init__(
            self,
            answer_key: dict[str, dict],
            field_names: Iterable[str] | None = None,
            match_threshold: float = 0.85,
    ):
        self.answer_key = answer_key
        self.field_names = _field_names_or_default(field_names)
        self.match_threshold = match_threshold
        self.total_processado = 0
        self.total_avaliavel = 0
        self.total_acertos = 0
        self.total_sem_gabarito = 0
        self.soma_similaridade = 0.0

    def evaluate_result(self, result: dict) -> dict:
        result["avaliacao"] = evaluate_extraction(
            image_path=result["image_path"],
            extracted_fields=_extracted_fields(result, self.field_names),
            answer_key=self.answer_key,
            field_names=self.field_names,
            match_threshold=self.match_threshold,
        )
        self._add_evaluation(result["avaliacao"])
        return result

    def evaluate_results(self, results: Iterable[dict]) -> Iterator[dict]:
        for result in results:
            yield self.evaluate_result(result)

    def summary(self) -> dict:
        acuracia = (
            self.total_acertos / self.total_avaliavel
            if self.total_avaliavel
            else 0.0
        )
        similaridade_media = (
            self.soma_similaridade / self.total_avaliavel
            if self.total_avaliavel
            else 0.0
        )

        return {
            "total_processado": self.total_processado,
            "total_avaliavel": self.total_avaliavel,
            "total_acertos": self.total_acertos,
            "total_erros": self.total_avaliavel - self.total_acertos,
            "total_sem_gabarito": self.total_sem_gabarito,
            "acuracia": round(acuracia, 4),
            "acuracia_percentual": round(acuracia * 100, 2),
            "similaridade_media": round(similaridade_media, 4),
            "campos_avaliados": self.field_names,
        }

    def _add_evaluation(self, evaluation: dict) -> None:
        self.total_processado += 1

        if evaluation["erro_avaliacao"] is not None:
            self.total_sem_gabarito += 1
            return

        self.total_avaliavel += 1
        self.total_acertos += int(evaluation["match"])
        self.soma_similaridade += evaluation.get("similaridade") or 0.0


def load_answer_key(answer_key_path: str | Path) -> dict[str, dict]:
    answer_key_path = Path(answer_key_path)
    content = answer_key_path.read_text(encoding="utf-8")

    match = re.search(r"const\s+answer\s*=\s*(\[.*\])", content, re.DOTALL)
    if match is None:
        raise ValueError(f"Formato de answer key não suportado: {answer_key_path}")

    answers = json.loads(match.group(1))
    return {answer["arquivo"]: answer for answer in answers}


def evaluate_extraction(
        image_path: str | Path,
        extracted_fields: dict,
        answer_key: dict[str, dict],
        field_names: Iterable[str] | None = None,
        match_threshold: float = 0.85,
) -> dict:
    field_names = _field_names_or_default(field_names)
    file_name = Path(image_path).name
    expected = answer_key.get(file_name)

    if expected is None:
        field_results = {
            field_name: {
                "esperado": None,
                "extraido": _field_value(extracted_fields, field_name),
                "similaridade": None,
                "distancia_levenshtein": None,
                "match": False,
            }
            for field_name in field_names
        }
        return _with_legacy_fields(
            {
                "arquivo": file_name,
                "campos": field_results,
                "similaridade": None,
                "distancia_levenshtein": None,
                "match": False,
                "erro_avaliacao": "Arquivo não encontrado no answer_key.",
            },
            field_names,
        )

    field_results = {}
    similarities = []

    for field_name in field_names:
        expected_value = _field_value(expected, field_name)
        extracted_value = _field_value(extracted_fields, field_name)
        similarity, distance = text_similarity(extracted_value, expected_value)

        field_results[field_name] = {
            "esperado": expected_value,
            "extraido": extracted_value,
            "similaridade": similarity,
            "distancia_levenshtein": distance,
            "match": similarity >= match_threshold,
        }
        similarities.append(similarity)

    average_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    return _with_legacy_fields(
        {
            "arquivo": file_name,
            "campos": field_results,
            "similaridade": round(average_similarity, 4),
            "distancia_levenshtein": None,
            "match": bool(field_results) and all(
                field["match"] for field in field_results.values()
            ),
            "erro_avaliacao": None,
        },
        field_names,
    )


def evaluate_course_extraction(
        image_path: str | Path,
        extracted_course: str | None,
        answer_key: dict[str, dict],
) -> dict:
    return evaluate_extraction(
        image_path=image_path,
        extracted_fields={"curso": extracted_course},
        answer_key=answer_key,
        field_names=["curso"],
    )


def text_similarity(value: str | None, expected: str | None) -> tuple[float, int | None]:
    if value is None or expected is None:
        return (1.0, 0) if value == expected else (0.0, None)

    normalized_value = normalize_text(value)
    normalized_expected = normalize_text(expected)

    if normalized_value == normalized_expected:
        return 1.0, 0

    max_len = max(len(normalized_value), len(normalized_expected))
    if max_len == 0:
        return 1.0, 0

    distance = levenshtein_distance(normalized_value, normalized_expected)
    similarity = 1 - (distance / max_len)

    return round(similarity, 4), distance


def course_similarity(value: str | None, expected: str | None) -> tuple[float, int | None]:
    return text_similarity(value, expected)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )
    return " ".join(without_accents.casefold().split())


def _field_value(fields: dict, field_name: str) -> object | None:
    if field_name in fields:
        return fields[field_name]

    normalized_field_name = normalize_text(field_name)

    for key, value in fields.items():
        if normalize_text(str(key)) == normalized_field_name:
            return value

    return None


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0

    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))

    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]

        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current_row[right_index - 1] + 1
            delete_cost = previous_row[right_index] + 1
            replace_cost = previous_row[right_index - 1] + (left_char != right_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))

        previous_row = current_row

    return previous_row[-1]


def _field_names_or_default(field_names: Iterable[str] | None) -> list[str]:
    if field_names is None:
        return DEFAULT_FIELD_NAMES.copy()

    if isinstance(field_names, str):
        field_names = [field_names]
        return field_names

    field_names = list(field_names)
    if not field_names:
        raise ValueError("field_names não pode ser vazio.")

    return field_names


def _extracted_fields(result: dict, field_names: Iterable[str]) -> dict:
    fields = result.get("fields")
    if isinstance(fields, dict):
        return fields

    fields = result.get("campos")
    if isinstance(fields, dict):
        return fields

    return {field_name: result.get(field_name) for field_name in field_names}


def _with_legacy_fields(evaluation: dict, field_names: list[str]) -> dict:
    if len(field_names) == 1:
        field = evaluation["campos"].get(field_names[0], {})
        evaluation["esperado"] = field.get("esperado")
        evaluation["extraido"] = field.get("extraido")
        evaluation["distancia_levenshtein"] = field.get("distancia_levenshtein")
        return evaluation

    evaluation["esperado"] = {
        field_name: field.get("esperado")
        for field_name, field in evaluation["campos"].items()
    }
    evaluation["extraido"] = {
        field_name: field.get("extraido")
        for field_name, field in evaluation["campos"].items()
    }
    return evaluation
