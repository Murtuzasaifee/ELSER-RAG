import hashlib
import structlog
import tiktoken

from elser_rag.config import settings
from elser_rag.models import Chunk, ParsedElement

logger = structlog.get_logger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _token_overlap_prefix(text: str, overlap_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    return _enc.decode(tokens[-overlap_tokens:])


def chunk_document(
    doc_id: str,
    elements: list[ParsedElement],
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    max_tokens = max_tokens or settings.max_chunk_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    log = logger.bind(doc_id=doc_id)
    log.debug(
        "chunking_start",
        element_count=len(elements),
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunks: list[Chunk] = []
    current_section = "Introduction"
    current_texts: list[str] = []
    current_pages: list[int] = []
    current_tokens = 0

    def flush(section: str) -> None:
        nonlocal current_texts, current_pages, current_tokens
        if not current_texts:
            return
        text = "\n".join(current_texts)
        chunk_index = len(chunks)
        chunk_id = _make_chunk_id(doc_id, chunk_index)
        page_start = min(current_pages)
        page_end = max(current_pages)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section_title=section,
                text=text,
                page_start=page_start,
                page_end=page_end,
                token_count=current_tokens,
            )
        )
        log.debug(
            "chunk_flushed",
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            section=section,
            token_count=current_tokens,
            page_start=page_start,
            page_end=page_end,
            text_preview=text[:60],
        )
        overlap_text = _token_overlap_prefix(text, overlap_tokens)
        overlap_token_count = _count_tokens(overlap_text) if overlap_text else 0
        log.debug(
            "overlap_carried",
            chunk_index=chunk_index,
            overlap_tokens=overlap_token_count,
        )
        current_texts = [overlap_text] if overlap_text else []
        current_pages = [current_pages[-1]]
        current_tokens = overlap_token_count

    for el in elements:
        if el.element_type == "title":
            log.debug("section_boundary", new_section=el.text, prev_section=current_section)
            flush(current_section)
            current_section = el.text
            continue

        el_tokens = _count_tokens(el.text)
        log.debug(
            "element_added",
            element_type=el.element_type,
            tokens=el_tokens,
            current_total=current_tokens,
            page=el.page_num,
        )

        if current_tokens + el_tokens > max_tokens and current_texts:
            log.debug(
                "overflow_flush",
                current_tokens=current_tokens,
                incoming_tokens=el_tokens,
                max_tokens=max_tokens,
            )
            flush(current_section)

        current_texts.append(el.text)
        current_pages.append(el.page_num)
        current_tokens += el_tokens

    flush(current_section)

    log.info("chunking_complete", chunk_count=len(chunks))
    return chunks


def _make_chunk_id(doc_id: str, index: int) -> str:
    raw = f"{doc_id}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
