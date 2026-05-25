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
    """Parse PDF into structured elements preserving document order."""
    path = Path(pdf_path)
    logger.info("parsing_pdf", path=str(path), filename=path.name)

    raw_elements = partition_pdf(
        filename=str(path),
        strategy="fast",
        include_page_breaks=False,
    )

    elements: list[ParsedElement] = []
    for el in raw_elements:
        text = el.text.strip() if el.text else ""
        if not text:
            continue

        page_num = _get_page_num(el)
        element_type = _classify(el)
        elements.append(ParsedElement(element_type=element_type, text=text, page_num=page_num))

    logger.info("pdf_parsed", filename=path.name, element_count=len(elements))
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
