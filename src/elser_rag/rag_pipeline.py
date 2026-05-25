import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog
from elasticsearch import AsyncElasticsearch

from elser_rag.chunker import chunk_document
from elser_rag.config import settings
from elser_rag.enricher import Enricher
from elser_rag.es_index import ESIndex
from elser_rag.generator import Generator
from elser_rag.models import DocumentRecord, IngestResult, QueryResult
from elser_rag.pdf_parser import parse_pdf
from elser_rag.retriever import Retriever

logger = structlog.get_logger(__name__)


class RAGPipeline:
    def __init__(self) -> None:
        self._es_index = ESIndex()
        self._enricher = Enricher()
        self._generator = Generator()
        self._retriever: Retriever | None = None
        logger.debug("rag_pipeline_initialized")

    async def setup(self) -> None:
        logger.info("rag_pipeline_setup_start")
        t0 = time.perf_counter()
        await self._es_index.setup()
        self._retriever = Retriever(self._es_index._es)
        elapsed = time.perf_counter() - t0
        logger.info("rag_pipeline_ready", setup_elapsed_s=round(elapsed, 3))

    # ------------------------------------------------------------------ #
    # Ingest                                                               #
    # ------------------------------------------------------------------ #

    async def ingest_document(self, pdf_path: str | Path) -> IngestResult:
        path = Path(pdf_path)
        doc_id = _doc_id_from_path(path)
        log = logger.bind(doc_id=doc_id, filename=path.name)
        log.info("ingest_start")
        log.debug("doc_id_derived", filename=path.name, doc_id=doc_id)

        # Stage 1: Parse
        t0 = time.perf_counter()
        elements = parse_pdf(path)
        parse_elapsed = time.perf_counter() - t0
        log.info("stage_parse_complete", element_count=len(elements), elapsed_s=round(parse_elapsed, 3))

        # Stage 2: Chunk
        t1 = time.perf_counter()
        chunks = chunk_document(doc_id=doc_id, elements=elements)
        chunk_elapsed = time.perf_counter() - t1
        log.info("stage_chunk_complete", chunk_count=len(chunks), elapsed_s=round(chunk_elapsed, 3))

        # Stage 3: Enrich
        t2 = time.perf_counter()
        enriched_chunks = await self._enricher.enrich_chunks(doc_id=doc_id, chunks=chunks)
        enrich_elapsed = time.perf_counter() - t2
        log.info("stage_enrich_complete", enriched_count=len(enriched_chunks), elapsed_s=round(enrich_elapsed, 3))

        # Stage 4: Index
        t3 = time.perf_counter()
        doc_summary = self._enricher._summary_cache.get(doc_id, "")
        record = DocumentRecord(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=len(enriched_chunks),
            doc_summary=doc_summary,
        )
        log.debug("indexing_document_record", chunk_count=len(enriched_chunks), summary_len=len(doc_summary))
        await self._es_index.index_document(record)
        await self._es_index.index_chunks(enriched_chunks, filename=path.name)
        index_elapsed = time.perf_counter() - t3
        log.info("stage_index_complete", elapsed_s=round(index_elapsed, 3))

        total_elapsed = time.perf_counter() - t0
        log.info(
            "ingest_complete",
            chunk_count=len(enriched_chunks),
            total_elapsed_s=round(total_elapsed, 3),
            stage_breakdown={
                "parse_s": round(parse_elapsed, 3),
                "chunk_s": round(chunk_elapsed, 3),
                "enrich_s": round(enrich_elapsed, 3),
                "index_s": round(index_elapsed, 3),
            },
        )
        return IngestResult(doc_id=doc_id, filename=path.name, chunk_count=len(enriched_chunks))

    async def ingest_directory(self, dir_path: str | Path) -> list[IngestResult]:
        directory = Path(dir_path)
        pdfs = sorted(directory.glob("*.pdf"))
        logger.info("ingest_directory_start", path=str(directory), pdf_count=len(pdfs))
        logger.debug("ingest_directory_files", files=[p.name for p in pdfs])

        results: list[IngestResult] = []
        for i, pdf in enumerate(pdfs):
            logger.info("ingest_directory_progress", file=pdf.name, progress=f"{i + 1}/{len(pdfs)}")
            try:
                result = await self.ingest_document(pdf)
                results.append(result)
            except Exception:
                logger.exception("ingest_document_failed", filename=pdf.name)

        logger.info("ingest_directory_complete", success_count=len(results), total=len(pdfs), failed=len(pdfs) - len(results))
        return results

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    async def query(self, text: str, top_k: int | None = None) -> QueryResult:
        if self._retriever is None:
            raise RuntimeError("Pipeline not initialized — call setup() first")

        top_k = top_k or settings.bm25_top_k
        log = logger.bind(query=text[:80], top_k=top_k)
        log.info("query_start")

        t0 = time.perf_counter()
        chunks = await self._retriever.retrieve(query=text, top_k=top_k)
        retrieve_elapsed = time.perf_counter() - t0
        log.info("query_retrieve_complete", chunks_retrieved=len(chunks), elapsed_s=round(retrieve_elapsed, 3))

        t1 = time.perf_counter()
        result = await self._generator.generate(query=text, chunks=chunks)
        generate_elapsed = time.perf_counter() - t1
        total_elapsed = time.perf_counter() - t0

        log.info(
            "query_complete",
            chunks_used=result.chunks_used,
            sources_count=len(result.sources),
            total_elapsed_s=round(total_elapsed, 3),
            stage_breakdown={
                "retrieve_s": round(retrieve_elapsed, 3),
                "generate_s": round(generate_elapsed, 3),
            },
        )
        return result

    # ------------------------------------------------------------------ #
    # Document management                                                  #
    # ------------------------------------------------------------------ #

    async def delete_document(self, doc_id: str) -> int:
        logger.info("delete_document_start", doc_id=doc_id)
        deleted = await self._es_index.delete_document(doc_id)
        logger.info("delete_document_complete", doc_id=doc_id, chunks_deleted=deleted)
        return deleted

    async def list_documents(self):
        logger.debug("list_documents_called")
        docs = await self._es_index.list_documents()
        logger.debug("list_documents_result", count=len(docs))
        return docs

    async def health(self) -> dict:
        logger.debug("health_check_called")
        return await self._es_index.health()

    async def close(self) -> None:
        logger.info("rag_pipeline_shutdown")
        await self._es_index.close()


def _doc_id_from_path(path: Path) -> str:
    return hashlib.sha256(path.name.encode()).hexdigest()[:16]
