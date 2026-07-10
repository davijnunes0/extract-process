from pathlib import Path


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


def format_summary(summary: dict) -> str:
    accuracy = summary.get("acuracia_percentual", 0)
    similarity = summary.get("similaridade_media", 0)

    lines = [
        _line("="),
        f"{Color.BOLD}{Color.CYAN}Resumo final{Color.RESET}",
        f"Processados: {summary.get('total_processado', 0)}",
        f"Avaliaveis: {summary.get('total_avaliavel', 0)}",
        f"Acertos: {Color.GREEN}{summary.get('total_acertos', 0)}{Color.RESET}",
        f"Erros: {Color.RED}{summary.get('total_erros', 0)}{Color.RESET}",
        f"Sem gabarito: {Color.YELLOW}{summary.get('total_sem_gabarito', 0)}{Color.RESET}",
        f"Acuracia: {_color_percent(accuracy)}",
        f"Similaridade media: {_color_percent(similarity * 100)}",
    ]

    fields = summary.get("campos_avaliados")
    if fields:
        lines.append(f"Campos avaliados: {', '.join(fields)}")

    lines.append(_line("="))
    return "\n".join(lines)


def _format_fields(field_results: dict) -> list[str]:
    lines = []

    for field_name, field in field_results.items():
        field_match = field.get("match")
        field_status = f"{Color.GREEN}[OK]{Color.RESET}" if field_match else f"{Color.RED}[ERRO]{Color.RESET}"

        lines.extend(
            [
                f"{field_status} {Color.BOLD}{field_name}{Color.RESET}",
                f"  {Color.CYAN}Extraido:{Color.RESET} {_value(field.get('extraido'))}",
                f"  {Color.BLUE}Esperado:{Color.RESET} {_value(field.get('esperado'))}",
            ]
        )

        similarity = field.get("similaridade")
        distance = field.get("distancia_levenshtein")

        if similarity is not None:
            lines.append(
                f"  {Color.YELLOW}Similaridade:{Color.RESET} "
                f"{similarity:.2%}  "
                f"{Color.GRAY}(distancia Levenshtein: {distance}){Color.RESET}"
            )

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
