# Architecture

## Two pipelines

### Ingest
```
PDF → pdf_parser.py → chunker.py → enricher.py → es_index.py
       (unstructured)  (section-    (OpenAI       (ES bulk +
        fast strategy)  aware,       context        ELSER ingest
                        512tok,      prefix)        pipeline)
                        50tok overlap)
```

### Query
```
Query → retriever.py → generator.py → QueryResult
         (ES hybrid      (OpenAI
          BM25 + ELSER    gpt-5.4
          RRF top-20)     + streaming)
```

## Modules

| File | Role |
|------|------|
| `config.py` | Pydantic Settings, all env vars, MeshAPI wired |
| `models.py` | ParsedElement, Chunk, EnrichedChunk, QueryResult, SourceRef, DocumentRecord, HealthStatus |
| `logging_config.py` | structlog JSON, third-party noise suppressed |
| `pdf_parser.py` | `partition_pdf` fast strategy → list of ParsedElement |
| `chunker.py` | Groups elements under section titles, flushes at 512 tokens, 50-token overlap carry-forward |
| `enricher.py` | Async OpenAI: doc summary (cached per doc_id, first 3 chunks) → per-chunk context prefix → `enriched_text = prefix + chunk` |
| `es_index.py` | ELSER model deployment, ingest pipeline setup, index creation, bulk indexing, CRUD, health |
| `retriever.py` | Hybrid RRF query: BM25 `multi_match` on `chunk_text`+`section_title^2` + `sparse_vector` on `enriched_text_elser`; source diversity cap (5 chunks/doc) |
| `generator.py` | Formats context with `[Source/Page/Section]` headers, OpenAI completion, async streaming |
| `rag_pipeline.py` | Orchestrator: `ingest_document`, `ingest_directory`, `query`, `delete_document` |
| `api.py` | FastAPI endpoints, lifecycle hooks, streaming response |

## Elasticsearch indices

**`elser_rag_docs`** — one doc per PDF
```
doc_id, filename, file_path, ingested_at, chunk_count, doc_summary
```

**`elser_rag_chunks`** — one doc per chunk
```
chunk_id, doc_id, filename, section_title (boost 2.0),
chunk_text (BM25 target), context_prefix (stored),
enriched_text, enriched_text_elser (sparse_vector ← ELSER ingest pipeline),
page_start, page_end, token_count
```

## ELSER flow
1. At startup: deploy `.elser-model-2-linux-x86_64`, create ingest pipeline `elser-rag-enrichment`
2. At index time: bulk insert hits ingest pipeline → processor calls ELSER inference on `enriched_text` → writes sparse tokens to `enriched_text_elser`
3. At query time: `sparse_vector` query sends raw text to same ELSER model → ES matches sparse tokens

## Why enrichment + ELSER together
ELSER handles semantic expansion but has no document-level context from an isolated chunk fragment. Context prefix bakes document identity + section topic into the text before ELSER encodes it → richer sparse vectors → better recall.
