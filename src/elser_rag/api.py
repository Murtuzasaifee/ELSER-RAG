import shutil
from pathlib import Path

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from elser_rag.config import settings
from elser_rag.models import DocumentRecord, HealthStatus, IngestResult, QueryResult
from elser_rag.rag_pipeline import RAGPipeline

logger = structlog.get_logger(__name__)

app = FastAPI(title="ELSER-RAG", version="0.1.0")

_pipeline: RAGPipeline | None = None


@app.on_event("startup")
async def startup() -> None:
    global _pipeline
    _pipeline = RAGPipeline()
    await _pipeline.setup()
    Path(settings.pdf_upload_dir).mkdir(parents=True, exist_ok=True)
    logger.info("api_started")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _pipeline:
        await _pipeline.close()


def _get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return _pipeline


# ------------------------------------------------------------------ #
# Ingest                                                               #
# ------------------------------------------------------------------ #


@app.post("/ingest", response_model=IngestResult, status_code=201)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResult:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    upload_path = Path(settings.pdf_upload_dir) / file.filename
    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = await _get_pipeline().ingest_document(upload_path)
    except Exception as exc:
        logger.exception("ingest_failed", filename=file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


class DirectoryIngestRequest(BaseModel):
    directory: str


@app.post("/ingest/directory", response_model=list[IngestResult])
async def ingest_directory(req: DirectoryIngestRequest) -> list[IngestResult]:
    path = Path(req.directory)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.directory}")
    try:
        return await _get_pipeline().ingest_directory(path)
    except Exception as exc:
        logger.exception("directory_ingest_failed", directory=req.directory)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------ #
# Query                                                                #
# ------------------------------------------------------------------ #


class QueryRequest(BaseModel):
    query: str
    top_k: int = 20
    stream: bool = False


@app.post("/query", response_model=QueryResult)
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    pipeline = _get_pipeline()

    if req.stream:
        chunks = await pipeline._retriever.retrieve(query=req.query, top_k=req.top_k)

        async def token_stream():
            async for token in pipeline._generator.generate_stream(req.query, chunks):
                yield token

        return StreamingResponse(token_stream(), media_type="text/plain")

    try:
        return await pipeline.query(text=req.query, top_k=req.top_k)
    except Exception as exc:
        logger.exception("query_failed", query=req.query[:80])
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------ #
# Document management                                                  #
# ------------------------------------------------------------------ #


@app.get("/documents", response_model=list[DocumentRecord])
async def list_documents() -> list[DocumentRecord]:
    return await _get_pipeline().list_documents()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    deleted = await _get_pipeline().delete_document(doc_id)
    return {"doc_id": doc_id, "chunks_deleted": deleted}


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #


@app.get("/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    try:
        detail = await _get_pipeline().health()
        es_status = detail.get("cluster_status", "unknown")
        elser_status = detail.get("elser_state", "unknown")
    except Exception as exc:
        return HealthStatus(elasticsearch="unreachable", elser_model="unknown", detail={"error": str(exc)})

    return HealthStatus(
        app="ok",
        elasticsearch=es_status,
        elser_model=elser_status,
        detail=detail,
    )


def main() -> None:
    import uvicorn
    uvicorn.run("elser_rag.api:app", host="0.0.0.0", port=8000, reload=False)
