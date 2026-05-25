# ELSER-RAG

Production-grade RAG over PDF documents using Elasticsearch ELSER — no vector database, no dense embeddings.

## What it does
Ingests PDFs → chunks → enriches each chunk with OpenAI context prefix → indexes into Elasticsearch where ELSER encodes enriched text as sparse vectors. At query time, hybrid BM25 + ELSER retrieval via RRF returns top-20 chunks → OpenAI generates final answer.

## Stack
- **Retrieval**: Elasticsearch 9.x ELSER (sparse vectors, `_inference` API) + BM25, fused via RRF
- **Enrichment**: OpenAI (MeshAPI) `openai/gpt-5.4-mini` — per-chunk context prefix at ingest time
- **Generation**: OpenAI (MeshAPI) `openai/gpt-5.4` — final answer with streaming support
- **PDF parsing**: `unstructured[pdf]` fast strategy — section-aware element classification
- **API**: FastAPI + uvicorn
- **Package manager**: `uv`

## Entry points
- `src/elser_rag/api.py` — FastAPI app, startup/shutdown lifecycle
- `src/elser_rag/rag_pipeline.py` — orchestrates ingest and query flows

## Run
```bash
cp .env.example .env   # fill OPENAI_API_KEY
make up-build          # docker compose up --build -d
```

## Key env vars
| Var | Default |
|-----|---------|
| `OPENAI_API_KEY` | required |
| `OPENAI_BASE_URL` | `https://api.meshapi.ai/v1` |
| `OPENAI_MODEL` | `openai/gpt-5.4` |
| `OPENAI_ENRICHMENT_MODEL` | `openai/gpt-5.4-mini` |
| `ELASTICSEARCH_URL` | `http://elasticsearch:9200` |
| `ELSER_MODEL_ID` | `.elser_model_2` (platform-agnostic; use `.elser_model_2_linux-x86_64` on native Linux x86) |
| `ELSER_INFERENCE_ID` | `elser-rag-inference` |

## Verification

```bash
# 1. Rebuild and start
make up-build

# 2. Check app is healthy (ELSER state should be "deployed")
curl http://localhost:8000/health | jq .

# 3. Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/test.pdf"

# 4. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

# 5. List ingested docs
curl http://localhost:8000/documents | jq .
```

> **Note:** First run downloads ELSER model (~70 MB) inside the ES container. `GET /health` may show `elser_state: "not_deployed"` for ~30–60s until download completes. Ingest will fail until ELSER is ready.

## API endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ingest` | Upload PDF |
| POST | `/ingest/directory` | Batch ingest from mounted path |
| POST | `/query` | Ask question, returns answer + sources |
| GET | `/documents` | List ingested docs |
| DELETE | `/documents/{doc_id}` | Remove doc + chunks |
| GET | `/health` | App + ES + ELSER status |
