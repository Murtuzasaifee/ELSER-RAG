from collections.abc import AsyncIterator

import structlog
from openai import AsyncOpenAI

from elser_rag.config import settings
from elser_rag.models import EnrichedChunk, QueryResult, SourceRef

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a precise document analyst. Answer the user's question using ONLY the provided document chunks.

Rules:
- Base your answer strictly on the provided context
- Cite sources by referencing filename and page numbers
- If the context does not contain enough information to answer, say so clearly
- Be concise and factual"""


def _format_context(chunks: list[EnrichedChunk]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[Source: {chunk.doc_id}, Pages: {chunk.page_start}-{chunk.page_end}, Section: {chunk.section_title}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _build_sources(chunks: list[EnrichedChunk]) -> list[SourceRef]:
    seen: set[str] = set()
    sources: list[SourceRef] = []
    for chunk in chunks:
        key = f"{chunk.doc_id}:{chunk.page_start}-{chunk.page_end}"
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceRef(
                    doc_id=chunk.doc_id,
                    filename=chunk.doc_id,
                    page_range=f"{chunk.page_start}-{chunk.page_end}",
                    section_title=chunk.section_title,
                )
            )
    return sources


class Generator:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    async def generate(self, query: str, chunks: list[EnrichedChunk]) -> QueryResult:
        context = _format_context(chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        response = await self._client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("answer_generated", query=query[:60], chunks_used=len(chunks))

        return QueryResult(
            answer=answer,
            sources=_build_sources(chunks),
            chunks_used=len(chunks),
            query=query,
        )

    async def generate_stream(
        self, query: str, chunks: list[EnrichedChunk]
    ) -> AsyncIterator[str]:
        context = _format_context(chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        stream = await self._client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
