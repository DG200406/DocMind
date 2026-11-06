from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from docmind.chat_engine import ChatEngine
from docmind.config import settings
from docmind.ingestion import load_and_chunk_file
from docmind.llm import get_llm
from docmind.reranker import Reranker
from docmind.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentsListResponse,
    UploadResponse,
)
from docmind.vectorstore import VectorStoreManager

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    settings.index_path.mkdir(parents=True, exist_ok=True)

    vectorstore = VectorStoreManager()
    reranker = Reranker()
    llm = get_llm()
    state["vectorstore"] = vectorstore
    state["engine"] = ChatEngine(vectorstore=vectorstore, reranker=reranker, llm=llm)
    yield
    state.clear()


app = FastAPI(title="DocMind", description="Ask questions about your documents.", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    dest = settings.docs_path / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    chunks = load_and_chunk_file(dest)
    added = state["vectorstore"].add_documents(chunks)
    return UploadResponse(filename=file.filename, chunks_added=added)


@app.get("/documents", response_model=DocumentsListResponse)
def list_documents():
    vectorstore: VectorStoreManager = state["vectorstore"]
    return DocumentsListResponse(
        documents=vectorstore.list_sources(), chunk_count=vectorstore.chunk_count()
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    engine: ChatEngine = state["engine"]
    return engine.ask(question=request.question, session_id=request.session_id)


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
    engine: ChatEngine = state["engine"]
    engine.reset_session(session_id)
    return {"status": "cleared", "session_id": session_id}
