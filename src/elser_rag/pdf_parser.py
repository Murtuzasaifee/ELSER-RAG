import structlog
from pathlib import Path

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title,
    NarrativeText,
    Text,
    Table,
    ListItem,
    Header,
)

from elser_rag.models import ParsedElement

logger = structlog.get_logger(__name__)


def parse_pdf(pdf_path: str | Path) -> list[ParsedElement]:
    path = Path(pdf_path)
    log = logger.bind(filename=path.name, path=str(path))
    log.info("parsing_pdf_start")

    raw_elements = partition_pdf(
        filename=str(path),
        strategy="fast",
        include_page_breaks=False,
    )

    log.debug("raw_elements_extracted", raw_count=len(raw_elements))

    elements: list[ParsedElement] = []
    skipped = 0
    type_counts: dict[str, int] = {}

    for el in raw_elements:
        text = el.text.strip() if el.text else ""
        if not text:
            skipped += 1
            continue

        page_num = _get_page_num(el)
        element_type = _classify(el)
        type_counts[element_type] = type_counts.get(element_type, 0) + 1

        log.debug(
            "element_classified",
            element_type=element_type,
            page=page_num,
            text_preview=text[:80],
            text_len=len(text),
        )
        elements.append(ParsedElement(element_type=element_type, text=text, page_num=page_num))

    log.info(
        "pdf_parsed",
        element_count=len(elements),
        skipped_empty=skipped,
        by_type=type_counts,
    )
    return elements


def _get_page_num(element) -> int:
    try:
        return element.metadata.page_number or 1
    except AttributeError:
        return 1


def _classify(element) -> str:
    if isinstance(element, (Title, Header)):
        return "title"
    if isinstance(element, Table):
        return "table"
    if isinstance(element, ListItem):
        return "list"
    return "text"
