import hashlib
import uuid
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

    async def setup(self) -> None:
        """Deploy ELSER, create ES indices. Call once at startup."""
        await self._es_index.setup()
        self._retriever = Retriever(self._es_index._es)
        logger.info("rag_pipeline_ready")

    # ------------------------------------------------------------------ #
    # Ingest                                                               #
    # ------------------------------------------------------------------ #

    async def ingest_document(self, pdf_path: str | Path) -> IngestResult:
        path = Path(pdf_path)
        doc_id = _doc_id_from_path(path)
        log = logger.bind(doc_id=doc_id, filename=path.name)
        log.info("ingest_start")

        elements = parse_pdf(path)
        chunks = chunk_document(doc_id=doc_id, elements=elements)
        enriched_chunks = await self._enricher.enrich_chunks(doc_id=doc_id, chunks=chunks)

        doc_summary = self._enricher._summary_cache.get(doc_id, "")
        record = DocumentRecord(
            doc_id=doc_id,
            filename=path.name,
            file_path=str(path),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            chunk_count=len(enriched_chunks),
            doc_summary=doc_summary,
        )
        await self._es_index.index_document(record)
        await self._es_index.index_chunks(enriched_chunks, filename=path.name)

        log.info("ingest_complete", chunk_count=len(enriched_chunks))
        return IngestResult(doc_id=doc_id, filename=path.name, chunk_count=len(enriched_chunks))

    async def ingest_directory(self, dir_path: str | Path) -> list[IngestResult]:
        directory = Path(dir_path)
        pdfs = sorted(directory.glob("*.pdf"))
        logger.info("ingest_directory_start", path=str(directory), pdf_count=len(pdfs))

        results: list[IngestResult] = []
        for pdf in pdfs:
            try:
                result = await self.ingest_document(pdf)
                results.append(result)
            except Exception:
                logger.exception("ingest_document_failed", filename=pdf.name)

        logger.info("ingest_directory_complete", success_count=len(results), total=len(pdfs))
        return results

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    async def query(self, text: str, top_k: int | None = None) -> QueryResult:
        if self._retriever is None:
            raise RuntimeError("Pipeline not initialized — call setup() first")

        top_k = top_k or settings.bm25_top_k
        logger.info("query_start", query=text[:80], top_k=top_k)

        chunks = await self._retriever.retrieve(query=text, top_k=top_k)
        result = await self._generator.generate(query=text, chunks=chunks)

        logger.info("query_complete", chunks_used=result.chunks_used)
        return result

    # ------------------------------------------------------------------ #
    # Document management                                                  #
    # ------------------------------------------------------------------ #

    async def delete_document(self, doc_id: str) -> int:
        return await self._es_index.delete_document(doc_id)

    async def list_documents(self):
        return await self._es_index.list_documents()

    async def health(self) -> dict:
        return await self._es_index.health()

    async def close(self) -> None:
        await self._es_index.close()


def _doc_id_from_path(path: Path) -> str:
    return hashlib.sha256(path.name.encode()).hexdigest()[:16]
