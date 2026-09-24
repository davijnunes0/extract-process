import json
from pathlib import Path


RAW_RESPONSE_LOG_LIMIT = 2000


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def format_extraction_log(
    result: dict,
    index: int,
    label: str | None = None,
) -> str:
    """Formata o resultado antes da comparação com o gabarito."""
    fields = result.get("fields")
    if not isinstance(fields, dict):
        fields = result.get("campos")
    if not isinstance(fields, dict):
        fields = {}

    extraction_label = str(
        label or result.get("label") or result.get("task") or "extração"
    )
    file_name = (
        result.get("document_name")
        or result.get("documento")
        or Path(str(result.get("image_path") or "")).name
        or "documento não informado"
    )
    lines = [
        _line(),
        (f"{Color.BOLD}[{extraction_label}] {index} {file_name}{Color.RESET}"),
    ]

    if _has_signature_pipeline_details(result, fields):
        lines.extend(_format_signature_extraction(result, fields))
    else:
        lines.extend(_format_generic_extraction(result, fields))

    error = result.get("error") or result.get("erro")
    crop_error = result.get("signature_crop_error")
    metadata_error = result.get("signature_metadata_error")

    if error:
        lines.append(f"{Color.RED}Erro:{Color.RESET} {error}")
    if crop_error and str(crop_error) not in str(error or ""):
        lines.append(f"{Color.RED}Erro no recorte:{Color.RESET} {crop_error}")
    if metadata_error:
        lines.append(
            f"{Color.YELLOW}Aviso nos metadados:{Color.RESET} {metadata_error}"
        )

    raw_response = str(result.get("raw_response") or result.get("resposta_bruta") or "")
    if error and raw_response:
        if len(raw_response) > RAW_RESPONSE_LOG_LIMIT:
            raw_response = (
                f"{raw_response[:RAW_RESPONSE_LOG_LIMIT]}\n... [resposta truncada]"
            )
        lines.append(f"{Color.GRAY}Resposta bruta:{Color.RESET}")
        lines.extend(f"    {line}" for line in raw_response.splitlines())

    return "\n".join(lines)


def _has_signature_pipeline_details(result: dict, fields: dict) -> bool:
    return bool(
        result.get("signature_crop_path")
        or result.get("signature_evidence")
        or result.get("verification_model")
        or "candidatos" in fields
    )


def format_result(result: dict) -> str:
    evaluation = result.get("avaliacao", {})
    match = evaluation.get("match")
    extraction_error = result.get("erro")
    evaluation_error = evaluation.get("erro_avaliacao")
    error = extraction_error or evaluation_error

    status = _status_label(
        match=match,
        extraction_error=extraction_error,
        evaluation_error=evaluation_error,
    )
    file_name = Path(result.get("image_path", "")).name

    lines = [
        _line(),
        f"{status} {Color.BOLD}{file_name}{Color.RESET}",
    ]

    field_results = evaluation.get("campos")
    if isinstance(field_results, dict) and field_results:
        lines.extend(_format_fields(field_results))
    else:
        lines.extend(_format_legacy_result(result, evaluation))

    if error:
        lines.append(f"{Color.RED}Erro:{Color.RESET} {error}")

    raw_response = result.get("resposta_bruta")
    if not match and raw_response:
        lines.append(f"{Color.GRAY}Resposta bruta:{Color.RESET} {raw_response}")

    return "\n".join(lines)


def _format_signature_extraction(result: dict, fields: dict) -> list[str]:
    decision = fields.get("assinaturas")
    document_type = fields.get("tipo_documento")
    region_is_valid = fields.get("regiao_valida")
    primary_role = fields.get("cargo_avaliado")
    role_category = fields.get("categoria_cargo")
    signature_type = fields.get("tipo_assinatura")
    localization_model = result.get("model")
    verification_model = result.get("verification_model")

    lines = [
        f"{Color.BOLD}Resultado{Color.RESET}",
        (f"  Assinatura: {_signature_decision_value(decision)}"),
        f"  Documento: {_plain_value(document_type)}",
        (f"  Região válida: {_boolean_value(region_is_valid)}"),
    ]

    if primary_role is not None:
        primary = str(primary_role)
        if role_category is not None:
            primary = f"{primary} ({role_category})"
        lines.append(f"  Cargo principal: {primary}")
    if signature_type is not None:
        lines.append(f"  Tipo: {signature_type}")

    visible_roles = fields.get("cargos_visiveis")
    if isinstance(visible_roles, (list, tuple)) and visible_roles:
        lines.append(f"  Cargos localizados: {', '.join(map(str, visible_roles))}")

    if verification_model is not None and verification_model != localization_model:
        lines.extend(
            [
                (f"  Modelo de localização: {_plain_value(localization_model)}"),
                (f"  Modelo de verificação: {_plain_value(verification_model)}"),
            ]
        )
    else:
        lines.append(
            f"  Modelo: {_plain_value(verification_model or localization_model)}"
        )

    lines.append(
        f"  Revisão de falso negativo: "
        f"{_review_value(result.get('false_negative_reviewed'))}"
    )
    lines.extend(_format_signature_candidates(fields.get("candidatos")))
    lines.extend(_format_signature_evidence(result, fields))
    return lines


def _format_generic_extraction(result: dict, fields: dict) -> list[str]:
    model = result.get("model")
    serialized_fields = json.dumps(
        fields,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    lines = [
        f"  Modelo: {_plain_value(model)}",
        f"{Color.BOLD}Campos extraídos{Color.RESET}",
    ]
    lines.extend(f"  {line}" for line in serialized_fields.splitlines())
    return lines


def _format_signature_candidates(value: object) -> list[str]:
    candidates = (
        [candidate for candidate in value if isinstance(candidate, dict)]
        if isinstance(value, (list, tuple))
        else []
    )
    lines = [f"{Color.BOLD}Candidatos verificados{Color.RESET}"]

    if not candidates:
        lines.append(f"  {Color.GRAY}Nenhum candidato autorizado.{Color.RESET}")
        return lines

    for candidate_index, candidate in enumerate(candidates, start=1):
        status = _candidate_status(candidate.get("assinatura_visual_detectada"))
        role = _plain_value(candidate.get("cargo"))
        details = [
            str(detail)
            for detail in (
                candidate.get("categoria_cargo"),
                candidate.get("tipo_assinatura"),
            )
            if detail is not None
        ]
        suffix = f" — {' | '.join(details)}" if details else ""
        if candidate.get("associacao_inferida") is True:
            suffix = f"{suffix} — associação inferida"

        lines.append(f"  {candidate_index}. {status} {role}{suffix}")
        verified_text = candidate.get("trecho_verificado")
        if verified_text:
            _append_multiline_value(
                lines,
                label="Trecho",
                value=str(verified_text),
                indent=5,
            )

    return lines


def _format_signature_evidence(result: dict, fields: dict) -> list[str]:
    evidence = result.get("signature_evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    crop_path = result.get("signature_crop_path") or fields.get("recorte_assinatura")
    lines = [f"{Color.BOLD}Evidência{Color.RESET}"]
    if crop_path is None:
        lines.append(f"  {Color.GRAY}Recorte não gerado.{Color.RESET}")
        return lines

    lines.append(f"  Recorte: {crop_path}")
    suggested = fields.get("coordenadas_sugeridas")
    applied = fields.get("coordenadas_recorte")
    if suggested is not None or applied is not None:
        lines.append(
            f"  Coordenadas (/1000): "
            f"{_plain_value(suggested)} -> {_plain_value(applied)}"
        )

    metadata_path = evidence.get("metadados")
    if metadata_path is not None:
        lines.append(f"  Metadados: {metadata_path}")
    return lines


def _append_multiline_value(
    lines: list[str],
    label: str,
    value: str,
    indent: int,
) -> None:
    value_lines = value.splitlines() or [""]
    prefix = " " * indent
    lines.append(f"{prefix}{label}: {value_lines[0]}")
    continuation = " " * (indent + len(label) + 2)
    lines.extend(f"{continuation}{line}" for line in value_lines[1:])


def _candidate_status(value: object) -> str:
    if value is True:
        return f"{Color.GREEN}[ASSINADO]{Color.RESET}"
    if value is False:
        return f"{Color.RED}[SEM ASSINATURA]{Color.RESET}"
    return f"{Color.YELLOW}[INCONCLUSIVO]{Color.RESET}"


def _signature_decision_value(value: object) -> str:
    normalized = str(value).strip().upper() if value is not None else ""
    if value is True or normalized == "TRUE":
        return f"{Color.GREEN}TRUE{Color.RESET}"
    if value is False or normalized == "FALSE":
        return f"{Color.RED}FALSE{Color.RESET}"
    return f"{Color.YELLOW}null{Color.RESET}"


def _boolean_value(value: object) -> str:
    if value is True:
        return f"{Color.GREEN}SIM{Color.RESET}"
    if value is False:
        return f"{Color.RED}NÃO{Color.RESET}"
    return f"{Color.YELLOW}INCONCLUSIVA{Color.RESET}"


def _review_value(value: object) -> str:
    if value is True:
        return f"{Color.YELLOW}SIM{Color.RESET}"
    if value is False:
        return "NÃO"
    return f"{Color.GRAY}NÃO SE APLICA{Color.RESET}"


def _plain_value(value: object) -> str:
    if value is None:
        return f"{Color.GRAY}não informado{Color.RESET}"
    return str(value)


def format_summary(summary: dict) -> str:
    accuracy = summary.get("acuracia_percentual", 0)
    similarity = summary.get("similaridade_media")

    lines = [
        _line("="),
        f"{Color.BOLD}{Color.CYAN}Resumo final{Color.RESET}",
        f"Processados: {summary.get('total_processado', 0)}",
        f"Avaliaveis: {summary.get('total_avaliavel', 0)}",
        f"Acertos: {Color.GREEN}{summary.get('total_acertos', 0)}{Color.RESET}",
        f"Erros: {Color.RED}{summary.get('total_erros', 0)}{Color.RESET}",
        f"Sem gabarito: {Color.YELLOW}{summary.get('total_sem_gabarito', 0)}{Color.RESET}",
        f"Acuracia: {_color_percent(accuracy)}",
    ]

    if similarity is not None:
        lines.append(f"Similaridade media: {_color_percent(similarity * 100)}")

    fields = summary.get("campos_avaliados")
    if fields:
        lines.append(f"Campos avaliados: {', '.join(fields)}")

    lines.append(_line("="))
    return "\n".join(lines)


def _format_fields(field_results: dict) -> list[str]:
    lines = []

    for field_name, field in field_results.items():
        field_match = field.get("match")
        field_status = (
            f"{Color.GREEN}[OK]{Color.RESET}"
            if field_match
            else f"{Color.RED}[ERRO]{Color.RESET}"
        )

        lines.extend(
            [
                f"{field_status} {Color.BOLD}{field_name}{Color.RESET}",
                f"  {Color.CYAN}Extraido:{Color.RESET} {_value(field.get('extraido'))}",
                f"  {Color.BLUE}Esperado:{Color.RESET} {_value(field.get('esperado'))}",
            ]
        )

        similarity = field.get("similaridade")
        distance = field.get("distancia_levenshtein")
        comparison_method = field.get("metodo_comparacao")

        if similarity is not None:
            lines.append(
                f"  {Color.YELLOW}Similaridade:{Color.RESET} "
                f"{similarity:.2%}  "
                f"{Color.GRAY}(distancia Levenshtein: {distance}){Color.RESET}"
            )
        elif comparison_method == "igualdade_exata":
            lines.append(f"  {Color.GRAY}Comparacao: igualdade exata{Color.RESET}")

    return lines


def _format_legacy_result(result: dict, evaluation: dict) -> list[str]:
    lines = [
        f"{Color.CYAN}Extraido:{Color.RESET} {_value(result.get('curso'))}",
        f"{Color.BLUE}Esperado:{Color.RESET} {_value(evaluation.get('esperado'))}",
    ]

    similarity = evaluation.get("similaridade")
    distance = evaluation.get("distancia_levenshtein")

    if similarity is not None:
        lines.append(
            f"{Color.YELLOW}Similaridade:{Color.RESET} "
            f"{similarity:.2%}  "
            f"{Color.GRAY}(distancia Levenshtein: {distance}){Color.RESET}"
        )

    return lines


def _status_label(
    match: bool | None,
    extraction_error: str | None,
    evaluation_error: str | None,
) -> str:
    if extraction_error:
        return f"{Color.RED}[ERRO]{Color.RESET}"

    if evaluation_error:
        return f"{Color.YELLOW}[SEM AVALIACAO]{Color.RESET}"

    if match:
        return f"{Color.GREEN}[OK]{Color.RESET}"

    return f"{Color.RED}[ERRO]{Color.RESET}"


def _value(value: object) -> str:
    if value is None:
        return f"{Color.GRAY}null{Color.RESET}"

    return str(value)


def _color_percent(value: float) -> str:
    if value >= 85:
        color = Color.GREEN
    elif value >= 60:
        color = Color.YELLOW
    else:
        color = Color.RED

    return f"{color}{value:.2f}%{Color.RESET}"


def _line(char: str = "-") -> str:
    return f"{Color.GRAY}{char * 72}{Color.RESET}"
