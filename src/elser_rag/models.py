from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParsedElement(BaseModel):
    element_type: Literal["title", "text", "table", "list"]
    text: str
    page_num: int


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section_title: str
    text: str
    page_start: int
    page_end: int
    token_count: int


class EnrichedChunk(Chunk):
    context_prefix: str
    enriched_text: str  # context_prefix + "\n" + text


class SourceRef(BaseModel):
    doc_id: str
    filename: str
    page_range: str
    section_title: str


class QueryResult(BaseModel):
    answer: str
    sources: list[SourceRef]
    chunks_used: int
    query: str


class IngestResult(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str = "success"


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    file_path: str
    ingested_at: str
    chunk_count: int
    doc_summary: str = ""


class HealthStatus(BaseModel):
    app: str = "ok"
    elasticsearch: str
    elser_model: str
    detail: dict = Field(default_factory=dict)
