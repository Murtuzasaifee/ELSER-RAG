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
        log = logger.bind(doc_id=doc_id)
        log.info("enrichment_start", total_chunks=len(chunks), model=settings.openai_enrichment_model)

        doc_summary = await self._get_doc_summary(doc_id, chunks)
        log.debug("doc_summary_ready", summary_len=len(doc_summary), summary_preview=doc_summary[:120])

        enriched: list[EnrichedChunk] = []
        for i, chunk in enumerate(chunks):
            log.info(
                "enriching_chunk",
                progress=f"{i + 1}/{len(chunks)}",
                chunk_id=chunk.chunk_id,
                section=chunk.section_title,
                token_count=chunk.token_count,
            )
            enriched_chunk = await self._enrich_chunk(chunk, doc_summary)
            log.debug(
                "chunk_enriched",
                chunk_id=chunk.chunk_id,
                prefix_len=len(enriched_chunk.context_prefix),
                enriched_text_len=len(enriched_chunk.enriched_text),
                prefix_preview=enriched_chunk.context_prefix[:100],
            )
            enriched.append(enriched_chunk)

        log.info("enrichment_complete", total_enriched=len(enriched))
        return enriched

    async def _get_doc_summary(self, doc_id: str, chunks: list[Chunk]) -> str:
        if doc_id in self._summary_cache:
            logger.debug("doc_summary_cache_hit", doc_id=doc_id)
            return self._summary_cache[doc_id]

        sample_chunks = chunks[:3]
        sample_text = "\n\n".join(c.text for c in sample_chunks)
        logger.debug(
            "doc_summary_request",
            doc_id=doc_id,
            sample_chunks=len(sample_chunks),
            sample_text_len=len(sample_text),
            model=settings.openai_enrichment_model,
            max_tokens=300,
        )

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
            logger.debug(
                "doc_summary_response",
                doc_id=doc_id,
                summary_len=len(summary),
                usage=response.usage.model_dump() if response.usage else None,
            )
        except Exception:
            logger.exception("doc_summary_failed", doc_id=doc_id)
            summary = ""

        self._summary_cache[doc_id] = summary
        logger.info("doc_summary_generated", doc_id=doc_id, summary_len=len(summary))
        return summary

    async def _enrich_chunk(self, chunk: Chunk, doc_summary: str) -> EnrichedChunk:
        prompt = _CONTEXT_PROMPT.format(
            doc_summary=doc_summary,
            section_title=chunk.section_title,
            chunk_text=chunk.text,
        )
        logger.debug(
            "chunk_enrichment_request",
            chunk_id=chunk.chunk_id,
            section=chunk.section_title,
            prompt_len=len(prompt),
            model=settings.openai_enrichment_model,
            max_tokens=150,
        )

        try:
            response = await self._client.chat.completions.create(
                model=settings.openai_enrichment_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            context_prefix = response.choices[0].message.content.strip()
            logger.debug(
                "chunk_enrichment_response",
                chunk_id=chunk.chunk_id,
                prefix_len=len(context_prefix),
                usage=response.usage.model_dump() if response.usage else None,
            )
        except Exception:
            logger.exception("chunk_enrichment_failed", chunk_id=chunk.chunk_id)
            context_prefix = ""

        enriched_text = f"{context_prefix}\n{chunk.text}" if context_prefix else chunk.text
        return EnrichedChunk(
            **chunk.model_dump(),
            context_prefix=context_prefix,
            enriched_text=enriched_text,
        )
