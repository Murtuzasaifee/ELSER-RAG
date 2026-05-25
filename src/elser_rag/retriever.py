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

        resp = await self._es.search(index=self._chunks_index, body=body)
        hits = resp["hits"]["hits"]

        chunks = [_hit_to_chunk(h) for h in hits]
        chunks = _apply_source_diversity(chunks)

        logger.info("retrieved_chunks", query=query[:60], count=len(chunks))
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
