import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
import pytesseract
from pytesseract import Output


SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
ROLE_HIERARCHY_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "assinaturas"
    / "hierarquia_cargos.json"
)


class ImageValidationError(ValueError):
    """Indica que uma imagem não pôde ser carregada ou validada."""


class SignatureLocalizationError(RuntimeError):
    """Indica que a região de assinatura não pôde ser localizada."""


@dataclass(frozen=True)
class RoleHierarchyEntry:
    """Cargo canônico, seus termos de âncora e sua prioridade na hierarquia."""

    role: str
    terms: frozenset[str]
    priority: int


@lru_cache(maxsize=None)
def load_signature_role_hierarchy(
    config_path: str | Path | None = None,
) -> tuple[RoleHierarchyEntry, ...]:
    """Carrega, do JSON configurado, a hierarquia de cargos institucionais.

    A hierarquia fica fora do código em ``hierarquia_cargos.json``: a ordem
    das entradas define a prioridade, e os termos são normalizados (sem
    acentos ou pontuação) antes de serem comparados com o OCR.

    Args:
        config_path: Caminho alternativo do JSON, usado em testes. Quando
            omitido, usa ``ROLE_HIERARCHY_CONFIG_PATH``.

    Returns:
        Tupla ordenada de cargos, do maior para o menor na hierarquia.

    Raises:
        SignatureLocalizationError: Se o arquivo estiver ausente, for um
            JSON inválido ou não seguir o formato esperado.
    """
    path = Path(config_path) if config_path is not None else ROLE_HIERARCHY_CONFIG_PATH
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SignatureLocalizationError(
            f"Não foi possível carregar a hierarquia de cargos de {path}: {error}"
        ) from error

    entries = raw_config.get("hierarquia") if isinstance(raw_config, dict) else None
    if not isinstance(entries, list) or not entries:
        raise SignatureLocalizationError(
            f"A hierarquia de cargos em {path} deve conter uma lista "
            "'hierarquia' não vazia."
        )

    hierarchy: list[RoleHierarchyEntry] = []
    seen_roles: set[str] = set()
    seen_terms: set[str] = set()
    for priority, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SignatureLocalizationError(
                f"Entrada {priority} da hierarquia em {path} deve ser um objeto."
            )
        role = entry.get("cargo")
        terms = entry.get("termos")
        if not isinstance(role, str) or not role.strip():
            raise SignatureLocalizationError(
                f"Entrada {priority} da hierarquia em {path} precisa de 'cargo'."
            )
        if (
            not isinstance(terms, list)
            or not terms
            or not all(isinstance(term, str) and term.strip() for term in terms)
        ):
            raise SignatureLocalizationError(
                f"Cargo {role!r} em {path} precisa de uma lista 'termos' não vazia."
            )
        role = role.strip()
        if role in seen_roles:
            raise SignatureLocalizationError(
                f"Cargo duplicado na hierarquia de {path}: {role}."
            )
        normalized_terms = frozenset(_normalize_ocr_token(term) for term in terms)
        repeated_terms = normalized_terms & seen_terms
        if repeated_terms:
            raise SignatureLocalizationError(
                f"Termos duplicados na hierarquia de {path}: "
                f"{', '.join(sorted(repeated_terms))}."
            )
        seen_roles.add(role)
        seen_terms.update(normalized_terms)
        hierarchy.append(
            RoleHierarchyEntry(role=role, terms=normalized_terms, priority=priority)
        )
    return tuple(hierarchy)


def signature_anchor_terms() -> frozenset[str]:
    """Retorna todos os termos de âncora definidos na hierarquia configurada."""
    return frozenset(
        term for entry in load_signature_role_hierarchy() for term in entry.terms
    )


@dataclass(frozen=True)
class OcrWord:
    """Palavra/expressão, caixa e linha do OCR no espaço da imagem original.

    ``line_id`` inclui a região lida para não misturar linhas de tentativas
    diferentes. O valor opcional mantém compatibilidade com chamadas antigas.
    """

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    line_id: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SignatureRegion:
    """Agrupa o recorte localizado, suas coordenadas e palavras-âncora.

    ``anchors`` contém somente o cargo selecionado; ``candidates`` reúne os
    cargos encontrados, inclusive os de menor hierarquia. A geometria do OCR
    delimita o bloco provável, mas não garante uma única assinatura manuscrita.
    Sem âncoras, a região é apenas evidência não localizada (fallback).
    """

    image: Image.Image
    box: tuple[int, int, int, int]
    anchors: tuple[OcrWord, ...]
    candidates: tuple[OcrWord, ...] = ()


@dataclass(frozen=True)
class SignatureOcrResult:
    """Agrupa o texto e os cargos reconhecidos no recorte original."""

    text: str
    anchor_terms: tuple[str, ...]


def load_and_validate_image(image_path: str | Path) -> Image.Image:
    """Carrega e valida uma imagem, respeitando sua orientação EXIF.

    A imagem retornada fica independente do arquivo aberto e conserva seu
    modo de cor, suas dimensões e sua qualidade original.

    Args:
        image_path: Caminho do arquivo de imagem que será carregado.

    Returns:
        Uma imagem PIL totalmente carregada e com orientação corrigida.

    Raises:
        ImageValidationError: Se o caminho for inválido, a extensão não for
            suportada ou o conteúdo não representar uma imagem válida.
    """
    path = Path(image_path).expanduser()

    if not path.exists():
        raise ImageValidationError(f"A imagem não existe: {path}")
    if not path.is_file():
        raise ImageValidationError(f"O caminho não é um arquivo: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ImageValidationError(
            f"Extensão não suportada: {path.suffix or '<sem extensão>'}. "
            f"Formatos aceitos: {supported}."
        )

    try:
        with Image.open(path) as source_image:
            source_image.load()
            oriented_image = ImageOps.exif_transpose(source_image)
            validated_image = oriented_image.copy()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ImageValidationError(
            f"Não foi possível carregar uma imagem válida de {path}: {error}"
        ) from error

    if validated_image.width <= 0 or validated_image.height <= 0:
        raise ImageValidationError(
            f"A imagem possui dimensões inválidas: {validated_image.size}."
        )

    return validated_image


def locate_and_crop_signature_region(
    image: Image.Image,
    language: str = "por",
    minimum_confidence: float = 20.0,
    minimum_anchor_y_ratio: float = 0.45,
    signature_height_ratio: float = 0.25,
    horizontal_margin_ratio: float = 0.15,
    lower_margin_ratio: float = 0.08,
    *,
    word_reader: Callable[[Image.Image, int, int], tuple[OcrWord, ...]] | None = None,
) -> SignatureRegion:
    """Localiza e recorta a assinatura de maior hierarquia do documento.

    Agrupa cargos compostos na mesma linha antes de aplicar a hierarquia.
    Se não encontra o cargo de maior prioridade, tenta regiões menores e
    sobrepostas, inclusive ao redor dos cargos já encontrados. Todas as
    leituras usam pixels originais, sem pré-processamento ou rotação.

    O recorte usa as linhas próximas de nome/cargo e limites entre blocos
    vizinhos. As proporções são tetos de folga, não margens fixas da página,
    e a folga busca incluir o traço manuscrito da assinatura e a rubrica
    junto ao nome/cargo. A busca é limitada: pode perder cargos, e não
    garante uma única assinatura manuscrita. As demais âncoras ficam em
    ``SignatureRegion.candidates``.

    Args:
        image: Imagem original já carregada e validada.
        language: Idioma instalado no Tesseract; ignorado com ``word_reader``.
        word_reader: Leitor alternativo chamado com imagem da região e offsets
            x/y. Deve retornar palavras/expressões no espaço da imagem original,
            com confiança 0–100 e identificadores de linha por região. Quando
            fornecido, dispensa o executável Tesseract. Falhas do leitor são
            propagadas, não interpretadas como ausência de âncora.
        minimum_confidence: Confiança mínima, entre 0 e 100, para palavras que
            não são âncoras exatas. Tokens de cargo reconhecidos como
            ``reitor``/``reitora`` e equivalentes são aceitos mesmo com
            confiança 0.
        minimum_anchor_y_ratio: Posição vertical mínima das âncoras, como
            fração da altura, usada para ignorar o texto introdutório.
        signature_height_ratio: Teto da folga acima do nome, como fração
            da altura total, reservada ao traço da assinatura; a folga
            efetiva depende da altura do texto.
        horizontal_margin_ratio: Teto da folga lateral relativo à largura,
            reservada à rubrica ao lado do nome/cargo.
        lower_margin_ratio: Teto da folga inferior relativo à altura,
            cobrindo a rubrica logo abaixo do nome/cargo.

    Returns:
        Região recortada ao redor da âncora selecionada, coordenadas em
        pixels, âncora selecionada e âncoras candidatas descartadas.

    Raises:
        ImageValidationError: Se a imagem possuir dimensões inválidas.
        ValueError: Se algum parâmetro estiver fora do intervalo permitido.
        SignatureLocalizationError: Se o Tesseract falhar ou se nenhuma
            âncora institucional autorizada for localizada.
    """
    _validate_image_dimensions(image)
    if word_reader is None:
        _validate_tesseract_language(language)
    _validate_ratio("minimum_anchor_y_ratio", minimum_anchor_y_ratio)
    _validate_ratio("signature_height_ratio", signature_height_ratio, allow_zero=False)
    _validate_ratio("horizontal_margin_ratio", horizontal_margin_ratio)
    _validate_ratio("lower_margin_ratio", lower_margin_ratio)
    if not 0 <= minimum_confidence <= 100:
        raise ValueError("minimum_confidence deve estar entre 0 e 100.")

    if word_reader is None:
        words = _extract_ocr_words_for_localization(image, language)
    else:
        search_top = round(image.height * 0.35)
        words = word_reader(
            image.crop((0, search_top, image.width, image.height)), 0, search_top
        )

    def candidates_for(scan_words: tuple[OcrWord, ...]) -> tuple[OcrWord, ...]:
        return tuple(
            word
            for word in _compound_anchors(scan_words)
            if _anchor_confidence_is_acceptable(word, minimum_confidence)
            and (word.top + word.height / 2) / image.height >= minimum_anchor_y_ratio
        )

    candidate_anchors = candidates_for(words)
    selected_anchor = _select_highest_priority_anchor(candidate_anchors)
    highest_role = load_signature_role_hierarchy()[0].role
    if (
        selected_anchor is None
        or canonical_anchor_role(selected_anchor.text) != highest_role
    ):
        for box in _localization_retry_boxes(
            image, candidate_anchors, minimum_anchor_y_ratio
        ):
            scan_words = (
                _ocr_words_from_prepared_image(
                    image.crop(box), language, box[0], box[1], 1.0, 0
                )
                if word_reader is None
                else word_reader(image.crop(box), box[0], box[1])
            )
            words += scan_words
            candidate_anchors = _merge_anchor_candidates(
                candidate_anchors + candidates_for(scan_words)
            )
        selected_anchor = _select_highest_priority_anchor(candidate_anchors)

    if selected_anchor is None:
        expected_terms = ", ".join(sorted(signature_anchor_terms()))
        raise SignatureLocalizationError(
            "Nenhuma âncora institucional autorizada foi localizada. "
            f"Termos esperados: {expected_terms}."
        )

    box = _calculate_anchor_based_box(
        anchor=selected_anchor,
        image_width=image.width,
        image_height=image.height,
        signature_height_ratio=signature_height_ratio,
        horizontal_margin_ratio=horizontal_margin_ratio,
        lower_margin_ratio=lower_margin_ratio,
        words=words,
        neighbors=candidate_anchors,
    )

    return SignatureRegion(
        image=image.crop(box),
        box=box,
        anchors=(selected_anchor,),
        candidates=candidate_anchors,
    )


def crop_fallback_signature_region(
    image: Image.Image,
    top_ratio: float = 0.72,
    horizontal_margin_ratio: float = 0.10,
) -> SignatureRegion:
    """Recorta a faixa inferior do documento quando nenhuma âncora foi encontrada.

    Evidência explicitamente não localizada: não seleciona cargo nem garante
    conter uma assinatura, muito menos uma única assinatura. Pode incluir
    vários signatários ou omitir todos. Não substitui a localização por OCR.
    Cobre aproximadamente o terço inferior com margens laterais da página.

    Args:
        image: Imagem original já carregada e validada.
        top_ratio: Fração da altura a partir da qual o recorte começa.
        horizontal_margin_ratio: Margem lateral relativa à largura total.

    Returns:
        Região inferior recortada, sem palavras-âncora associadas.

    Raises:
        ImageValidationError: Se a imagem possuir dimensões inválidas.
        ValueError: Se algum parâmetro estiver fora do intervalo permitido.
    """
    _validate_image_dimensions(image)
    _validate_ratio("top_ratio", top_ratio)
    _validate_ratio("horizontal_margin_ratio", horizontal_margin_ratio)
    if top_ratio >= 1:
        raise ValueError("top_ratio deve estar no intervalo [0, 1).")

    horizontal_margin = round(image.width * horizontal_margin_ratio)
    x_min = min(horizontal_margin, max(image.width - 1, 0))
    x_max = max(image.width - horizontal_margin, x_min + 1)
    y_min = min(round(image.height * top_ratio), image.height - 1)
    box = (x_min, y_min, x_max, image.height)
    return SignatureRegion(
        image=image.crop(box),
        box=box,
        anchors=(),
    )


def execute_ocr_on_signature_region(
    region_image: Image.Image,
    language: str = "por",
    page_segmentation_mode: int = 11,
) -> SignatureOcrResult:
    """Executa OCR uma vez no recorte original, sem pré-processamento.

    Confirma os cargos institucionais autorizados encontrados no recorte.

    O texto retornado corresponde principalmente aos rótulos impressos junto
    à assinatura. O Tesseract não é projetado para transcrever com precisão
    o traço manuscrito de uma assinatura.

    Args:
        region_image: Região previamente localizada e recortada.
        language: Idioma instalado que será usado pelo Tesseract.
        page_segmentation_mode: Modo de segmentação de página entre 3 e 13.

    Returns:
        Texto reconhecido e cargos institucionais encontrados no recorte.

    Raises:
        ImageValidationError: Se a região possuir dimensões inválidas.
        ValueError: Se o idioma ou modo de segmentação forem inválidos.
        SignatureLocalizationError: Se a execução do Tesseract falhar.
    """
    _validate_image_dimensions(region_image)
    _validate_tesseract_language(language)
    if (
        not isinstance(page_segmentation_mode, int)
        or isinstance(page_segmentation_mode, bool)
        or not 3 <= page_segmentation_mode <= 13
    ):
        raise ValueError("page_segmentation_mode deve estar entre 3 e 13.")

    config = f"--oem 3 --psm {page_segmentation_mode}"
    try:
        text = pytesseract.image_to_string(
            region_image,
            lang=language,
            config=config,
        ).strip()
    except pytesseract.TesseractError as error:
        raise SignatureLocalizationError(
            f"O Tesseract falhou durante o OCR do recorte: {error}"
        ) from error

    return SignatureOcrResult(
        text=text,
        anchor_terms=_find_authorized_anchor_terms(text),
    )


def crop_to_document_content(
    image: Image.Image,
    background_threshold: int = 245,
    padding_ratio: float = 0.02,
) -> tuple[Image.Image, tuple[int, int]]:
    """Remove margens claras e devolve o recorte com o deslocamento original.

    Args:
        image: Imagem original já carregada e validada.
        background_threshold: Intensidades iguais ou acima disso entram como
            fundo e são cortadas.
        padding_ratio: Folga proporcional recolocada ao redor do conteúdo.

    Returns:
        A imagem recortada no conteúdo e o deslocamento ``(left, top)`` em
        relação à imagem original.

    Raises:
        ImageValidationError: Se a imagem possuir dimensões inválidas.
        ValueError: Se o limiar ou a folga forem inválidos.
    """
    _validate_image_dimensions(image)
    if (
        not isinstance(background_threshold, int)
        or isinstance(background_threshold, bool)
        or not 0 <= background_threshold <= 255
    ):
        raise ValueError("background_threshold deve estar entre 0 e 255.")
    _validate_ratio("padding_ratio", padding_ratio)

    grayscale = convert_to_grayscale(image)
    lookup_table = [
        255 if intensity < background_threshold else 0 for intensity in range(256)
    ]
    content_box = grayscale.point(lookup_table).getbbox()
    if content_box is None:
        return image.copy(), (0, 0)

    left, top, right, bottom = content_box
    content_area = (right - left) * (bottom - top)
    if content_area < image.width * image.height * 0.05:
        return image.copy(), (0, 0)

    padding = round(min(image.size) * padding_ratio)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    if left >= right or top >= bottom:
        return image.copy(), (0, 0)
    return image.crop((left, top, right, bottom)), (left, top)


def _extract_ocr_words_for_localization(
    image: Image.Image,
    language: str,
) -> tuple[OcrWord, ...]:
    """Obtém palavras e coordenadas do Tesseract para localizar assinaturas.

    Lê a faixa inferior original, sem analisar fundo, converter cores,
    redimensionar ou rotacionar. As caixas usam o espaço da imagem original.
    """
    search_top = round(image.height * 0.35)
    search_region = image.crop((0, search_top, image.width, image.height))
    origin_x = 0
    origin_y = search_top

    return _ocr_words_from_prepared_image(
        search_region,
        language=language,
        offset_x=origin_x,
        offset_y=origin_y,
        scale_factor=1.0,
        border_size=0,
    )


def _ocr_words_from_prepared_image(
    ocr_image: Image.Image,
    language: str,
    offset_x: int,
    offset_y: int,
    scale_factor: float,
    border_size: int,
) -> tuple[OcrWord, ...]:
    """Lê palavras do Tesseract e mapeia as caixas para a imagem original."""
    try:
        data: dict[str, list[Any]] = pytesseract.image_to_data(
            ocr_image,
            lang=language,
            config="--oem 3 --psm 11",
            output_type=Output.DICT,
        )
    except pytesseract.TesseractError as error:
        raise SignatureLocalizationError(
            f"O Tesseract falhou durante a localização: {error}"
        ) from error

    words: list[OcrWord] = []
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
            left, top, width, height = _map_prepared_box_to_source(
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
                offset_x=offset_x,
                offset_y=offset_y,
                scale_factor=scale_factor,
                border_size=border_size,
            )
        except (TypeError, ValueError, IndexError) as error:
            raise SignatureLocalizationError(
                "O Tesseract retornou coordenadas inválidas."
            ) from error
        words.append(
            OcrWord(
                text=text,
                confidence=confidence,
                left=left,
                top=top,
                width=width,
                height=height,
                line_id=(
                    offset_x,
                    offset_y,
                    ocr_image.width,
                    ocr_image.height,
                    *(
                        int(data[key][index])
                        for key in ("page_num", "block_num", "par_num", "line_num")
                    ),
                )
                if all(
                    key in data
                    for key in ("page_num", "block_num", "par_num", "line_num")
                )
                else None,
            )
        )
    return tuple(words)


def _map_prepared_box_to_source(
    left: int,
    top: int,
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
    scale_factor: float,
    border_size: int,
) -> tuple[int, int, int, int]:
    """Converte uma caixa do OCR preparado para o espaço da imagem original."""
    if scale_factor <= 0:
        raise ValueError("scale_factor deve ser maior que zero.")
    mapped_left = round((left - border_size) / scale_factor) + offset_x
    mapped_top = round((top - border_size) / scale_factor) + offset_y
    mapped_width = max(1, round(width / scale_factor))
    mapped_height = max(1, round(height / scale_factor))
    return mapped_left, mapped_top, mapped_width, mapped_height


def _anchor_confidence_is_acceptable(
    word: OcrWord,
    minimum_confidence: float,
) -> bool:
    """Aceita âncora exata mesmo com confiança 0; demais usam o mínimo."""
    if _is_signature_anchor(word.text):
        return True
    return word.confidence >= minimum_confidence


def _word_lines(words: tuple[OcrWord, ...]) -> list[list[OcrWord]]:
    """Agrupa por linha Tesseract; usa geometria para chamadas legadas.

    Espaços grandes separam colunas mesmo com o mesmo identificador de linha.
    Os identificadores incluem a região de leitura.
    """
    groups: dict[tuple[int, ...], list[OcrWord]] = {}
    ungrouped: list[list[OcrWord]] = []
    for word in words:
        if word.line_id is not None:
            groups.setdefault(word.line_id, []).append(word)
        else:
            line = next(
                (
                    line
                    for line in ungrouped
                    if abs(line[0].top - word.top)
                    <= max(1, min(line[0].height, word.height) / 2)
                ),
                None,
            )
            if line is None:
                ungrouped.append([word])
            else:
                line.append(word)
    lines: list[list[OcrWord]] = []
    for group in list(groups.values()) + ungrouped:
        segment: list[OcrWord] = []
        for word in sorted(group, key=lambda item: item.left):
            if segment and word.left - (segment[-1].left + segment[-1].width) > 3 * max(
                word.height, segment[-1].height
            ):
                lines.append(segment)
                segment = []
            segment.append(word)
        if segment:
            lines.append(segment)
    return lines


def _joined_words(words: list[OcrWord]) -> OcrWord:
    """Une texto e caixas sem perder a extensão integral do cargo composto."""
    left = min(word.left for word in words)
    top = min(word.top for word in words)
    return OcrWord(
        " ".join(word.text for word in words),
        min(word.confidence for word in words),
        left,
        top,
        max(word.left + word.width for word in words) - left,
        max(word.top + word.height for word in words) - top,
        words[0].line_id,
    )


def _compound_anchors(words: tuple[OcrWord, ...]) -> tuple[OcrWord, ...]:
    """Reconhece o termo mais longo por linha, consumindo seus componentes.

    Não remove prefixos desconhecidos para promover um cargo. Prefixos
    separados por hífen também impedem aceitar isoladamente o componente.
    Expressões de OCR por linha são tokenizadas apenas para reconhecer cargos:
    cada token conserva a caixa inteira da expressão, sem estimar posições de
    palavras que o backend não forneceu.
    """
    terms = signature_anchor_terms()
    max_length = max(map(len, terms))
    anchors: list[OcrWord] = []
    for line in _word_lines(words):
        line = [
            replace(word, text=token) for word in line for token in word.text.split()
        ]
        index = 0
        while index < len(line):
            if not _normalize_ocr_token(line[index].text):
                index += 1
                continue
            match: tuple[int, OcrWord] | None = None
            for end in range(index + 1, len(line) + 1):
                text = " ".join(word.text for word in line[index:end])
                normalized = _normalize_ocr_token(text)
                if len(normalized) > max_length:
                    break
                if normalized in terms and _normalize_ocr_token(line[end - 1].text):
                    match = (end, _joined_words(line[index:end]))
            if match is not None:
                previous = line[index - 1].text if index else ""
                blocked = previous.endswith(("-", "–", "—")) or _normalize_ocr_token(
                    previous
                ) in {"vice", "pro", "ex", "sub"}
                if not blocked:
                    anchors.append(match[1])
                index = match[0]
            else:
                index += 1
    return tuple(anchors)


def _merge_anchor_candidates(anchors: tuple[OcrWord, ...]) -> tuple[OcrWord, ...]:
    """Deduplica leituras sobrepostas; compostos prevalecem sobre seus filhos."""
    merged: list[OcrWord] = []
    for anchor in sorted(
        anchors,
        key=lambda word: (
            -len(_normalize_ocr_token(word.text)),
            -word.confidence,
            word.height,
        ),
    ):
        if any(
            min(anchor.left + anchor.width, other.left + other.width)
            > max(anchor.left, other.left)
            and min(anchor.top + anchor.height, other.top + other.height)
            > max(anchor.top, other.top)
            for other in merged
        ):
            continue
        merged.append(anchor)
    return tuple(merged)


def _localization_retry_boxes(
    image: Image.Image,
    anchors: tuple[OcrWord, ...],
    minimum_y: float,
) -> tuple[tuple[int, int, int, int], ...]:
    """Busca limitada em faixas/colunas sobrepostas, sem alterar os pixels.

    Uma faixa na altura dos cargos conhecidos permite recuperar um superior
    omitido ao lado deles. Duas faixas cobrem o restante da área elegível.
    """
    start = max(0, round(image.height * minimum_y) - round(image.height * 0.04))
    span = image.height - start
    bands = [
        (start, start + round(span * 0.65)),
        (start + round(span * 0.35), image.height),
    ]
    if anchors:
        bands.insert(
            0,
            (
                max(0, min(w.top for w in anchors) - round(image.height * 0.10)),
                min(
                    image.height,
                    max(w.top + w.height for w in anchors) + round(image.height * 0.06),
                ),
            ),
        )
    boxes = []
    for top, bottom in bands:
        for left, right in (
            (0, image.width),
            (0, round(image.width * 0.60)),
            (round(image.width * 0.40), image.width),
        ):
            box = (left, top, right, bottom)
            if right > left and bottom > top and box not in boxes:
                boxes.append(box)
    return tuple(boxes)


def _signatory_text_box(
    anchor: OcrWord,
    words: tuple[OcrWord, ...],
    neighbors: tuple[OcrWord, ...],
) -> tuple[int, int, int, int]:
    """Estima o bloco cargo/nome pela linha do cargo e a linha próxima acima."""
    height = max(1, anchor.height)
    # Não una hipóteses de segmentação incompatíveis de leituras diferentes.
    if anchor.line_id is not None:
        words = tuple(
            word
            for word in words
            if word.line_id is not None and word.line_id[:4] == anchor.line_id[:4]
        )
    lines = [_joined_words(line) for line in _word_lines(words)]
    cargo = [anchor]
    for line in lines:
        if (
            abs(line.top - anchor.top) <= height / 2
            and line.left <= anchor.left + anchor.width
            and line.left + line.width >= anchor.left
            and line.height <= 2 * height
            and not any(
                other != anchor
                and line.left <= other.left
                and line.left + line.width >= other.left + other.width
                and abs(other.top - anchor.top) <= height
                for other in neighbors
            )
        ):
            cargo.append(line)
    block = _joined_words(cargo)
    above = [
        line
        for line in lines
        if 0 <= anchor.top - (line.top + line.height) <= 3 * height
        and line.height <= 2 * height
        and line.left < block.left + block.width
        and line.left + line.width > block.left
        and line.width <= max(block.width * 2, height * 24)
        and not any(
            other != anchor
            and line.left <= other.left < line.left + line.width
            and abs(line.top - other.top) <= height
            for other in neighbors
        )
    ]
    if above:
        name = min(
            above,
            key=lambda line: (
                anchor.top - line.top - line.height,
                abs(line.left + line.width / 2 - block.left - block.width / 2),
            ),
        )
        block = _joined_words([block, name])
    return block.left, block.top, block.left + block.width, block.top + block.height


def _calculate_anchor_based_box(
    anchor: OcrWord,
    image_width: int,
    image_height: int,
    signature_height_ratio: float,
    horizontal_margin_ratio: float,
    lower_margin_ratio: float,
    minimum_region_width_ratio: float = 0.18,
    words: tuple[OcrWord, ...] = (),
    neighbors: tuple[OcrWord, ...] = (),
) -> tuple[int, int, int, int]:
    """Recorta nome/cargo com folga local e limites entre blocos vizinhos.

    As proporções limitam folgas baseadas na altura do cargo, abrangendo o
    traço da assinatura acima e a rubrica ao lado ou logo abaixo do nome/cargo.
    A região nunca fica mais estreita que ``minimum_region_width_ratio`` da
    largura da imagem, centrada no bloco cargo/nome. Limites nunca cortam o
    cargo selecionado e respeitam a imagem. Blocos sobrepostos ou assinaturas
    não reconhecidas impedem garantir isolamento.
    """
    left, top, right, bottom = _signatory_text_box(anchor, words, neighbors)
    height = max(1, anchor.height)
    margin = min(round(image_width * horizontal_margin_ratio), 6 * height)
    x_min, x_max = left - margin, right + margin
    half_width = round(image_width * minimum_region_width_ratio / 2)
    center = (left + right) // 2
    x_min, x_max = min(x_min, center - half_width), max(x_max, center + half_width)
    y_min = top - min(round(image_height * signature_height_ratio), 8 * height)
    y_max = bottom + min(round(image_height * lower_margin_ratio), 3 * height)
    for other in neighbors:
        if other == anchor:
            continue
        ol, ot, oright, ob = _signatory_text_box(other, words, neighbors)
        if ot < bottom + 2 * height and ob > top - 2 * height:
            if oright <= left:
                x_min = max(x_min, (oright + left) // 2)
            elif ol >= right:
                x_max = min(x_max, (right + ol) // 2)
            elif other.left + other.width <= anchor.left:
                x_min = max(x_min, (other.left + other.width + anchor.left) // 2)
            elif other.left >= anchor.left + anchor.width:
                x_max = min(x_max, (anchor.left + anchor.width + other.left) // 2)
        if ol < right and oright > left:
            if ob <= top:
                y_min = max(y_min, (ob + top) // 2)
            elif ot >= bottom:
                y_max = min(y_max, (bottom + ot) // 2)
    return (
        max(0, min(x_min, anchor.left)),
        max(0, min(y_min, anchor.top)),
        min(image_width, max(x_max, anchor.left + anchor.width)),
        min(image_height, max(y_max, anchor.top + anchor.height)),
    )


def _select_highest_priority_anchor(
    anchors: tuple[OcrWord, ...],
) -> OcrWord | None:
    """Seleciona a âncora de maior hierarquia entre as candidatas.

    A hierarquia vem de ``hierarquia_cargos.json``. Quando há várias âncoras
    do mesmo cargo, vence a mais baixa na página; persistindo o empate, a de
    maior confiança de OCR.
    """
    if not anchors:
        return None

    hierarchy = load_signature_role_hierarchy()
    priority_by_term: dict[str, int] = {
        term: entry.priority for entry in hierarchy for term in entry.terms
    }
    fallback_priority = len(hierarchy)

    def sort_key(anchor: OcrWord) -> tuple[int, int, float]:
        priority = priority_by_term.get(
            _normalize_ocr_token(anchor.text), fallback_priority
        )
        bottom = anchor.top + anchor.height
        return (priority, -bottom, -anchor.confidence)

    return min(anchors, key=sort_key)


def _is_signature_anchor(text: str) -> bool:
    """Informa se uma palavra representa um cargo institucional autorizado."""
    normalized = _normalize_ocr_token(text)
    return normalized in signature_anchor_terms()


def _find_authorized_anchor_terms(text: str) -> tuple[str, ...]:
    """Retorna cargos por linha, sem promover componentes de títulos compostos."""
    words = tuple(
        OcrWord(token, 100, index * 20, row * 20, 20, 10, (row,))
        for row, line in enumerate(text.splitlines())
        for index, token in enumerate(line.split())
    )
    return tuple(
        dict.fromkeys(
            _normalize_ocr_token(word.text) for word in _compound_anchors(words)
        )
    )


def _normalize_ocr_token(text: str) -> str:
    """Remove acentos e pontuação de uma palavra reconhecida pelo OCR."""
    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z]", "", without_accents.lower())


def canonical_anchor_role(text: str) -> str | None:
    """Converte uma palavra-âncora no código de cargo da hierarquia configurada."""
    normalized = _normalize_ocr_token(text)
    for entry in load_signature_role_hierarchy():
        if normalized in entry.terms:
            return entry.role
    return None


def scale_to_thousand(value: int, total: int) -> int:
    """Normaliza um valor em pixels para a escala 0-1000 da extração."""
    if total <= 0:
        return 0
    return max(0, min(1000, round(value * 1000 / total)))


def normalize_pixel_box(
    box: tuple[int, int, int, int] | list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
    """Converte [x_min, y_min, x_max, y_max] de pixels para 0-1000."""
    x_min, y_min, x_max, y_max = (int(value) for value in box)
    return [
        scale_to_thousand(x_min, image_width),
        scale_to_thousand(y_min, image_height),
        scale_to_thousand(x_max, image_width),
        scale_to_thousand(y_max, image_height),
    ]


def serialize_ocr_anchor(
    word: OcrWord,
    image_width: int,
    image_height: int,
    crop_origin: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """Serializa uma âncora com texto, cargo e coordenadas para a IA."""
    box = [
        word.left,
        word.top,
        word.left + word.width,
        word.top + word.height,
    ]
    origin_x, origin_y = crop_origin
    box_recorte = [
        word.left - origin_x,
        word.top - origin_y,
        word.left + word.width - origin_x,
        word.top + word.height - origin_y,
    ]
    return {
        "text": word.text,
        "cargo": canonical_anchor_role(word.text),
        "x": word.left,
        "y": word.top,
        "box": box,
        "coordenadas": normalize_pixel_box(
            (box[0], box[1], box[2], box[3]),
            image_width,
            image_height,
        ),
        "x_recorte": box_recorte[0],
        "y_recorte": box_recorte[1],
        "box_recorte": box_recorte,
        "confidence": round(float(word.confidence), 2),
    }


def serialize_localization(
    region: SignatureRegion,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """Monta o payload de localização com caixa da assinatura e âncoras.

    ``anchors`` descreve somente a âncora selecionada, presente no recorte.
    ``candidatos`` lista todas as âncoras encontradas, marcando a selecionada,
    para que as assinaturas de menor hierarquia descartadas permaneçam
    rastreáveis no JSON de saída. Listas vazias de âncoras/cargos no fallback
    representam evidência não localizada, não uma assinatura confirmada.
    """
    crop_origin = (region.box[0], region.box[1])
    anchors = [
        serialize_ocr_anchor(
            word,
            image_width,
            image_height,
            crop_origin=crop_origin,
        )
        for word in region.anchors
    ]
    selected_words = set(region.anchors)
    candidates = [
        {
            **serialize_ocr_anchor(
                word,
                image_width,
                image_height,
                crop_origin=crop_origin,
            ),
            "selecionada": word in selected_words,
        }
        for word in region.candidates
    ]
    cargos: list[str] = []
    for anchor in anchors:
        cargo = anchor["cargo"]
        if isinstance(cargo, str) and cargo not in cargos:
            cargos.append(cargo)
    box = list(region.box)
    coordenadas_recorte = normalize_pixel_box(
        (box[0], box[1], box[2], box[3]),
        image_width,
        image_height,
    )
    return {
        "image_size": [image_width, image_height],
        "box": box,
        "coordenadas_recorte": coordenadas_recorte,
        "coordenadas_recorte_ia": [
            0,
            coordenadas_recorte[1],
            1000,
            coordenadas_recorte[3],
        ],
        "regiao_assinatura": {
            "box": box,
            "coordenadas_recorte": coordenadas_recorte,
            "size": [
                max(0, box[2] - box[0]),
                max(0, box[3] - box[1]),
            ],
        },
        "cargos_visiveis": cargos,
        "anchors": anchors,
        "candidatos": candidates,
        "localization_anchors": [word.text for word in region.anchors],
    }


def _validate_tesseract_language(language: str) -> None:
    """Valida a forma do identificador de idioma recebido pelo Tesseract."""
    if not isinstance(language, str) or not re.fullmatch(
        r"[A-Za-z0-9_.+-]+",
        language,
    ):
        raise ValueError("language contém um identificador inválido.")


def _validate_ratio(
    parameter_name: str,
    value: float,
    allow_zero: bool = True,
) -> None:
    """Valida um valor proporcional limitado ao intervalo entre zero e um."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value > 1
        or value < 0
        or (not allow_zero and value == 0)
    ):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{parameter_name} deve estar no intervalo {interval}.")


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """Converte uma imagem válida para escala de cinza.

    Args:
        image: Imagem PIL que será convertida.

    Returns:
        Uma nova imagem no modo ``L``, com oito bits por pixel.

    Raises:
        ImageValidationError: Se a imagem possuir dimensões inválidas.
    """
    _validate_image_dimensions(image)
    return ImageOps.grayscale(image)


def save_processing_image(
    image: Image.Image,
    output_path: str | Path,
) -> Path:
    """Salva uma imagem intermediária em PNG sem perdas.

    Args:
        image: Imagem intermediária que será salva.
        output_path: Destino obrigatório com extensão ``.png``.

    Returns:
        Caminho do arquivo PNG criado.

    Raises:
        ImageValidationError: Se a imagem possuir dimensões inválidas.
        ValueError: Se o destino não usar a extensão PNG.
        OSError: Se o diretório ou arquivo não puder ser criado.
    """
    _validate_image_dimensions(image)
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise ValueError("As imagens intermediárias devem ser salvas como PNG.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False)
    return destination


def _validate_image_dimensions(image: Image.Image) -> None:
    """Valida o tipo e as dimensões de uma imagem PIL."""
    if not isinstance(image, Image.Image):
        raise ImageValidationError("O valor informado não é uma imagem PIL.")
    if image.width <= 0 or image.height <= 0:
        raise ImageValidationError(
            f"A imagem possui dimensões inválidas: {image.size}."
        )
