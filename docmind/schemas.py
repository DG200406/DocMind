from typing import List, Optional

from pydantic import BaseModel


class SourceSnippet(BaseModel):
    source: str
    chunk_id: str
    score: float
    text: str


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceSnippet]
    latency_ms: float


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


class DocumentsListResponse(BaseModel):
    documents: List[str]
    chunk_count: int
