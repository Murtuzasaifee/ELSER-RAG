import structlog
from elasticsearch import AsyncElasticsearch, NotFoundError

from elser_rag.config import settings
from elser_rag.models import DocumentRecord, EnrichedChunk

logger = structlog.get_logger(__name__)

_DOCS_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id":       {"type": "keyword"},
            "filename":     {"type": "keyword"},
            "file_path":    {"type": "keyword"},
            "ingested_at":  {"type": "date"},
            "chunk_count":  {"type": "integer"},
            "doc_summary":  {"type": "text"},
        }
    }
}

_CHUNKS_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id":      {"type": "keyword"},
            "doc_id":        {"type": "keyword"},
            "filename":      {"type": "keyword"},
            "section_title": {"type": "text", "boost": 2.0},
            "chunk_text":    {"type": "text"},
            "context_prefix": {"type": "text", "index": False},
            "enriched_text": {"type": "text"},
            "enriched_text_elser": {
                "type": "sparse_vector"
            },
            "page_start":    {"type": "integer"},
            "page_end":      {"type": "integer"},
            "token_count":   {"type": "integer"},
        }
    }
}


class ESIndex:
    def __init__(self) -> None:
        self._es = AsyncElasticsearch(settings.elasticsearch_url)
        self._docs_index = settings.elasticsearch_docs_index
        self._chunks_index = settings.elasticsearch_chunks_index
        self._elser_model_id = settings.elser_model_id
        # Inference pipeline ID (without OS suffix)
        self._inference_id = ".elser-model-2"

    async def setup(self) -> None:
        """Create indices and deploy ELSER inference pipeline if not present."""
        await self._ensure_elser_deployed()
        await self._ensure_ingest_pipeline()
        await self._ensure_index(self._docs_index, _DOCS_MAPPING)
        await self._ensure_index(self._chunks_index, _CHUNKS_MAPPING)
        logger.info("es_setup_complete")

    async def _ensure_elser_deployed(self) -> None:
        try:
            resp = await self._es.ml.get_trained_models(model_id=self._elser_model_id)
            models = resp.get("trained_model_configs", [])
            if models:
                logger.info("elser_model_found", model_id=self._elser_model_id)
                await self._start_elser_deployment()
                return
        except Exception:
            pass

        logger.info("deploying_elser", model_id=self._elser_model_id)
        await self._es.ml.put_trained_model(
            model_id=self._elser_model_id,
            body={"input": {"field_names": ["text_field"]}},
        )
        await self._start_elser_deployment()

    async def _start_elser_deployment(self) -> None:
        try:
            await self._es.ml.start_trained_model_deployment(
                model_id=self._elser_model_id,
                wait_for="started",
            )
            logger.info("elser_deployed", model_id=self._elser_model_id)
        except Exception as exc:
            # Already started or allocation exists — not fatal
            logger.warning("elser_deploy_skipped", reason=str(exc))

    async def _ensure_ingest_pipeline(self) -> None:
        pipeline_id = "elser-rag-enrichment"
        try:
            await self._es.ingest.get_pipeline(id=pipeline_id)
            logger.debug("ingest_pipeline_exists", pipeline_id=pipeline_id)
            return
        except NotFoundError:
            pass

        await self._es.ingest.put_pipeline(
            id=pipeline_id,
            body={
                "description": "ELSER sparse encoding of enriched_text",
                "processors": [
                    {
                        "inference": {
                            "model_id": self._elser_model_id,
                            "input_output": [
                                {
                                    "input_field": "enriched_text",
                                    "output_field": "enriched_text_elser",
                                }
                            ],
                        }
                    }
                ],
            },
        )
        logger.info("ingest_pipeline_created", pipeline_id=pipeline_id)

    async def _ensure_index(self, index: str, mapping: dict) -> None:
        exists = await self._es.indices.exists(index=index)
        if not exists:
            await self._es.indices.create(index=index, body=mapping)
            logger.info("index_created", index=index)
        else:
            logger.debug("index_exists", index=index)

    async def index_document(self, record: DocumentRecord) -> None:
        await self._es.index(
            index=self._docs_index,
            id=record.doc_id,
            document=record.model_dump(),
        )
        logger.debug("document_indexed", doc_id=record.doc_id)

    async def index_chunks(self, chunks: list[EnrichedChunk], filename: str) -> None:
        if not chunks:
            return

        operations: list[dict] = []
        for chunk in chunks:
            operations.append({"index": {"_index": self._chunks_index, "_id": chunk.chunk_id}})
            operations.append(
                {
                    "chunk_id":      chunk.chunk_id,
                    "doc_id":        chunk.doc_id,
                    "filename":      filename,
                    "section_title": chunk.section_title,
                    "chunk_text":    chunk.text,
                    "context_prefix": chunk.context_prefix,
                    "enriched_text": chunk.enriched_text,
                    "page_start":    chunk.page_start,
                    "page_end":      chunk.page_end,
                    "token_count":   chunk.token_count,
                }
            )

        resp = await self._es.bulk(
            operations=operations,
            pipeline="elser-rag-enrichment",
        )
        if resp.get("errors"):
            failed = [i for i in resp["items"] if i.get("index", {}).get("error")]
            logger.error("bulk_index_errors", failed_count=len(failed), sample=failed[:2])
        else:
            logger.info("chunks_indexed", doc_id=chunks[0].doc_id, count=len(chunks))

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        try:
            resp = await self._es.get(index=self._docs_index, id=doc_id)
            return DocumentRecord(**resp["_source"])
        except NotFoundError:
            return None

    async def list_documents(self) -> list[DocumentRecord]:
        resp = await self._es.search(
            index=self._docs_index,
            body={"query": {"match_all": {}}, "size": 1000, "sort": [{"ingested_at": "desc"}]},
        )
        return [DocumentRecord(**hit["_source"]) for hit in resp["hits"]["hits"]]

    async def delete_document(self, doc_id: str) -> int:
        # Delete doc metadata
        try:
            await self._es.delete(index=self._docs_index, id=doc_id)
        except NotFoundError:
            pass

        # Delete all chunks for this doc
        resp = await self._es.delete_by_query(
            index=self._chunks_index,
            body={"query": {"term": {"doc_id": doc_id}}},
        )
        deleted = resp.get("deleted", 0)
        logger.info("document_deleted", doc_id=doc_id, chunks_deleted=deleted)
        return deleted

    async def health(self) -> dict:
        cluster = await self._es.cluster.health()
        try:
            model_resp = await self._es.ml.get_trained_models_stats(model_id=self._elser_model_id)
            deployment_state = (
                model_resp["trained_model_stats"][0]
                .get("deployment_stats", {})
                .get("state", "unknown")
            )
        except Exception:
            deployment_state = "not_deployed"
        return {
            "cluster_status": cluster["status"],
            "elser_state": deployment_state,
        }

    async def close(self) -> None:
        await self._es.close()
