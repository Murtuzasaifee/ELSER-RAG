import structlog
from elasticsearch import AsyncElasticsearch

from elser_rag.config import settings
from elser_rag.models import EnrichedChunk

logger = structlog.get_logger(__name__)

_MAX_CHUNKS_PER_DOC = 5


class Retriever:
    def __init__(self, es: AsyncElasticsearch) -> None:
        self._es = es
        self._chunks_index = settings.elasticsearch_chunks_index
        self._inference_id = settings.elser_inference_id

    async def retrieve(self, query: str, top_k: int | None = None) -> list[EnrichedChunk]:
        top_k = top_k or settings.bm25_top_k

        logger.info(
            "retrieval_start",
            query=query[:80],
            top_k=top_k,
            index=self._chunks_index,
            inference_id=self._inference_id,
        )

        body = {
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["chunk_text", "section_title^2"],
                                    }
                                }
                            }
                        },
                        {
                            "standard": {
                                "query": {
                                    "sparse_vector": {
                                        "field": "enriched_text_elser",
                                        "inference_id": self._inference_id,
                                        "query": query,
                                    }
                                }
                            }
                        },
                    ],
                    "rank_window_size": top_k,
                }
            },
            "size": top_k,
        }

        logger.debug(
            "es_query_body",
            retrievers=["bm25_multi_match(chunk_text,section_title^2)", f"elser_sparse_vector({self._inference_id})"],
            rank_window_size=top_k,
        )

        resp = await self._es.search(index=self._chunks_index, body=body)
        hits = resp["hits"]["hits"]

        logger.debug("raw_hits_received", raw_hit_count=len(hits))

        for i, hit in enumerate(hits):
            logger.debug(
                "hit_detail",
                rank=i + 1,
                chunk_id=hit["_source"].get("chunk_id"),
                doc_id=hit["_source"].get("doc_id"),
                section=hit["_source"].get("section_title", "")[:50],
                score=hit.get("_score"),
                page_start=hit["_source"].get("page_start"),
                page_end=hit["_source"].get("page_end"),
            )

        chunks = [_hit_to_chunk(h) for h in hits]
        chunks_before_diversity = len(chunks)
        chunks = _apply_source_diversity(chunks)
        dropped = chunks_before_diversity - len(chunks)

        logger.debug(
            "source_diversity_applied",
            before=chunks_before_diversity,
            after=len(chunks),
            dropped=dropped,
            max_per_doc=_MAX_CHUNKS_PER_DOC,
        )

        logger.info("retrieval_complete", query=query[:60], chunks_returned=len(chunks))
        return chunks


def _hit_to_chunk(hit: dict) -> EnrichedChunk:
    src = hit["_source"]
    return EnrichedChunk(
        chunk_id=src["chunk_id"],
        doc_id=src["doc_id"],
        section_title=src.get("section_title", ""),
        text=src["chunk_text"],
        page_start=src.get("page_start", 0),
        page_end=src.get("page_end", 0),
        token_count=src.get("token_count", 0),
        context_prefix=src.get("context_prefix", ""),
        enriched_text=src.get("enriched_text", src["chunk_text"]),
    )


def _apply_source_diversity(chunks: list[EnrichedChunk]) -> list[EnrichedChunk]:
    doc_counts: dict[str, int] = {}
    result: list[EnrichedChunk] = []
    for chunk in chunks:
        count = doc_counts.get(chunk.doc_id, 0)
        if count < _MAX_CHUNKS_PER_DOC:
            result.append(chunk)
            doc_counts[chunk.doc_id] = count + 1
    return result
