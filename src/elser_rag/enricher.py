import structlog
from openai import AsyncOpenAI

from elser_rag.config import settings
from elser_rag.models import Chunk, EnrichedChunk

logger = structlog.get_logger(__name__)

_SUMMARY_PROMPT = """\
You are a document analyst. Given the first portion of a document, write a concise 3-5 sentence summary covering:
- What type of document this is
- The main subject or topic
- The key entities, dates, or metrics mentioned

Respond with only the summary paragraph."""

_CONTEXT_PROMPT = """\
Document summary:
{doc_summary}

Chunk from section "{section_title}":
{chunk_text}

Write a 2-3 sentence context prefix that describes:
1. What document this chunk comes from and its overall topic
2. What specific aspect or subtopic this section covers
3. Why this information is relevant in context

Respond with only the context prefix sentences."""


class Enricher:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._summary_cache: dict[str, str] = {}

    async def enrich_chunks(self, doc_id: str, chunks: list[Chunk]) -> list[EnrichedChunk]:
        doc_summary = await self._get_doc_summary(doc_id, chunks)
        enriched: list[EnrichedChunk] = []
        for chunk in chunks:
            enriched_chunk = await self._enrich_chunk(chunk, doc_summary)
            enriched.append(enriched_chunk)
        logger.info("enriched_chunks", doc_id=doc_id, count=len(enriched))
        return enriched

    async def _get_doc_summary(self, doc_id: str, chunks: list[Chunk]) -> str:
        if doc_id in self._summary_cache:
            return self._summary_cache[doc_id]

        # Use first 3 chunks as document sample
        sample_text = "\n\n".join(c.text for c in chunks[:3])
        try:
            response = await self._client.chat.completions.create(
                model=settings.openai_enrichment_model,
                messages=[
                    {"role": "system", "content": _SUMMARY_PROMPT},
                    {"role": "user", "content": sample_text},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            summary = response.choices[0].message.content.strip()
        except Exception:
            logger.exception("doc_summary_failed", doc_id=doc_id)
            summary = ""

        self._summary_cache[doc_id] = summary
        logger.debug("doc_summary_generated", doc_id=doc_id, length=len(summary))
        return summary

    async def _enrich_chunk(self, chunk: Chunk, doc_summary: str) -> EnrichedChunk:
        prompt = _CONTEXT_PROMPT.format(
            doc_summary=doc_summary,
            section_title=chunk.section_title,
            chunk_text=chunk.text,
        )
        try:
            response = await self._client.chat.completions.create(
                model=settings.openai_enrichment_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            context_prefix = response.choices[0].message.content.strip()
        except Exception:
            logger.exception("chunk_enrichment_failed", chunk_id=chunk.chunk_id)
            context_prefix = ""

        enriched_text = f"{context_prefix}\n{chunk.text}" if context_prefix else chunk.text
        return EnrichedChunk(
            **chunk.model_dump(),
            context_prefix=context_prefix,
            enriched_text=enriched_text,
        )
