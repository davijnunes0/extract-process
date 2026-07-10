import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

from source.services.ai_client import AIClientError, AiClient
from source.services.image_service import image_to_base64, image_to_data_url


class ExtractField:
    def __init__(
            self,
            client: AiClient,
            image_path: str | Path,
            model: str,
            field_names: Iterable[str] | None = None,
    ):
        self.client = client
        self.image_path = Path(image_path)
        self.model = model
        self.field_names = list(field_names) if field_names is not None else None

    def extract(
            self,
            prompt: str | Path | None = None,
            field_names: Iterable[str] | None = None,
    ) -> dict:
        return self.extract_image(
            client=self.client,
            image_path=self.image_path,
            model=self.model,
            prompt=prompt,
            field_names=field_names or self.field_names,
        )

    def extract_many(
            self,
            image_paths: Iterable[str | Path],
            prompt: str | Path | None = None,
            field_names: Iterable[str] | None = None,
    ) -> Iterator[dict]:
        yield from self.extract_images(
            client=self.client,
            image_paths=image_paths,
            model=self.model,
            prompt=prompt,
            field_names=field_names or self.field_names,
        )

    @staticmethod
    def extract_image(
            client: AiClient,
            image_path: str | Path,
            model: str,
            prompt: str | Path | None = None,
            field_names: Iterable[str] | None = None,
    ) -> dict:
        image_path = Path(image_path)

        if isinstance(field_names, str):
            field_names = [field_names]
        elif field_names is not None:
            field_names = list(field_names)

        if isinstance(prompt, Path):
            prompt = prompt.read_text(encoding="utf-8")

        if prompt is None:
            prompt = "Extraia os campos solicitados da imagem e retorne JSON valido."

        messages = ExtractField._build_messages(client, image_path, prompt)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
        except AIClientError as exc:
            empty_fields = ExtractField._empty_fields(field_names)
            error_message = f"Erro ao chamar LLM: {exc}"
            raw_response = str(exc)

            return ExtractField._build_result(
                image_path=image_path,
                model=model,
                fields=empty_fields,
                raw_response=raw_response,
                error=error_message,
            )

        content = response.choices[0].message.content
        raw_response = ExtractField._raw_response_text(response, content)
        data = ExtractField._load_structured_response(content, field_names)

        if data is None:
            empty_fields = ExtractField._empty_fields(field_names)
            error_message = "Resposta da LLM nao veio em formato estruturado valido."

            return ExtractField._build_result(
                image_path=image_path,
                model=model,
                fields=empty_fields,
                raw_response=raw_response,
                error=error_message,
            )

        fields = ExtractField._extract_fields(data, field_names)

        return ExtractField._build_result(
            image_path=image_path,
            model=model,
            fields=fields,
            raw_response=raw_response,
            error=None,
        )

    @staticmethod
    def extract_images(
            client: AiClient,
            image_paths: Iterable[str | Path],
            model: str,
            prompt: str | Path | None = None,
            field_names: Iterable[str] | None = None,
    ) -> Iterator[dict]:
        for image_path in image_paths:
            result = ExtractField.extract_image(
                client=client,
                image_path=image_path,
                model=model,
                prompt=prompt,
                field_names=field_names,
            )
            result["image_path"] = str(image_path)
            result["document_name"] = Path(image_path).name
            yield result

    @staticmethod
    def extract_course_image(
            client: AiClient,
            image_path: str | Path,
            model: str,
            prompt: str | Path | None = None,
    ) -> dict:
        result = ExtractField.extract_image(
            client=client,
            image_path=image_path,
            model=model,
            prompt=prompt,
            field_names=["curso"],
        )
        result["curso"] = result["campos"].get("curso")
        return result

    @staticmethod
    def extract_course_images(
            client: AiClient,
            image_paths: Iterable[str | Path],
            model: str,
            prompt: str | Path | None = None,
    ) -> Iterator[dict]:
        for image_path in image_paths:
            result = ExtractField.extract_course_image(
                client=client,
                image_path=image_path,
                model=model,
                prompt=prompt,
            )
            result["image_path"] = str(image_path)
            result["document_name"] = Path(image_path).name
            yield result

    @staticmethod
    def _extract_fields(data: dict, field_names: list[str] | None) -> dict:
        source = data.get("campos") if isinstance(data.get("campos"), dict) else data

        if field_names is None:
            field_names = list(source.keys())

        normalized_source = {
            ExtractField._normalize_field_name(field_name): value
            for field_name, value in source.items()
        }

        fields = {}
        for field_name in field_names:
            value = source.get(field_name)

            if value is None:
                value = normalized_source.get(
                    ExtractField._normalize_field_name(field_name)
                )

            fields[field_name] = ExtractField._normalize_value(value)

        return fields

    @staticmethod
    def _normalize_field_name(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        without_accents = "".join(
            char for char in normalized
            if not unicodedata.combining(char)
        )
        return without_accents.casefold().strip()

    @staticmethod
    def _normalize_value(value: object) -> str | None:
        if value is None:
            return None

        value = str(value).strip()
        if value == "":
            return None

        normalized = value.casefold()
        null_values = {
            "null",
            "none",
            "n/a",
            "nao encontrado",
            "não encontrado",
            "nao encontrado.",
            "não encontrado.",
            "não visível",
            "nao visivel",
        }
        if normalized in null_values:
            return None

        return value

    @staticmethod
    def _empty_fields(field_names: list[str] | None) -> dict:
        if field_names is None:
            return {}

        return {field_name: None for field_name in field_names}

    @staticmethod
    def _build_result(
            image_path: Path,
            model: str,
            fields: dict,
            raw_response: str,
            error: str | None,
    ) -> dict:
        return {
            "document_name": image_path.name,
            "image_path": str(image_path),
            "model": model,
            "fields": fields,
            "raw_response": raw_response,
            "error": error,
            "campos": fields,
            "resposta_bruta": raw_response,
            "erro": error,
        }

    @staticmethod
    def _raw_response_text(response: object, content: str) -> str:
        if content:
            return content

        raw = getattr(response, "raw", None)
        if raw is None:
            return ""

        try:
            return json.dumps(raw, ensure_ascii=False)
        except TypeError:
            return str(raw)

    @staticmethod
    def _load_structured_response(
            content: str,
            field_names: list[str] | None = None,
    ) -> dict | None:
        data = ExtractField._load_json_response(content)
        if data is not None:
            return data

        markdown_data = ExtractField._load_markdown_fields(content, field_names)
        if markdown_data:
            return markdown_data

        return None

    @staticmethod
    def _load_json_response(content: str) -> dict | None:
        content = content.strip()
        fenced_json = ExtractField._extract_fenced_json(content)

        if fenced_json is not None:
            data = ExtractField._parse_json_object(fenced_json)
            if data is not None:
                return data

        data = ExtractField._parse_json_object(content)
        if data is not None:
            return data

        json_start = content.find("{")
        if json_start == -1:
            return None

        try:
            data, _ = json.JSONDecoder().raw_decode(content[json_start:])
        except json.decoder.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        return data

    @staticmethod
    def _parse_json_object(content: str) -> dict | None:
        try:
            data = json.loads(content)
        except json.decoder.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        return data

    @staticmethod
    def _extract_fenced_json(content: str) -> str | None:
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            return None

        return match.group(1)

    @staticmethod
    def _load_markdown_fields(
            content: str,
            field_names: list[str] | None,
    ) -> dict:
        if not field_names:
            return {}

        fields = {}

        for field_name in field_names:
            pattern = rf"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(field_name)}(?:\*\*)?\s*[:=-]\s*(.+?)\s*$"
            match = re.search(pattern, content)
            if match is None:
                continue

            value = match.group(1).strip()
            value = value.strip("`*_ ")
            fields[field_name] = value

        return fields

    @staticmethod
    def _build_messages(
            client: AiClient,
            image_path: str | Path,
            prompt: str,
    ) -> list[dict]:
        chat_path = getattr(getattr(client, "config", None), "chat_completions_path", "")

        if "ollama" in chat_path:
            return [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_to_base64(image_path)],
                }
            ]

        image_data_url = image_to_data_url(image_path)
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            }
        ]


ExtractCourse = ExtractField
