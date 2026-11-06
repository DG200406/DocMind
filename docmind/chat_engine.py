"""Conversational RAG: retrieve -> re-rank -> generate, with per-session memory."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from docmind.config import settings
from docmind.reranker import Reranker
from docmind.schemas import ChatResponse, SourceSnippet
from docmind.vectorstore import VectorStoreManager

MAX_HISTORY_TURNS = 6

QA_PROMPT = """You are DocMind, an assistant that answers questions using ONLY the \
context below, which was retrieved from the user's own documents. If the context \
does not contain the answer, say you don't have enough information in the documents.

Conversation so far:
{history}

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class Session:
    history: List[Tuple[str, str]] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.history.append((question, answer))
        if len(self.history) > MAX_HISTORY_TURNS:
            self.history.pop(0)

    def format(self) -> str:
        if not self.history:
            return "(no prior turns)"
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history)

    def last_question(self) -> str:
        return self.history[-1][0] if self.history else ""


class ChatEngine:
    def __init__(
        self,
        vectorstore: VectorStoreManager,
        reranker: Reranker,
        llm: BaseLanguageModel,
    ):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.llm = llm
        self.sessions: Dict[str, Session] = {}

    def _get_session(self, session_id: str) -> Session:
        return self.sessions.setdefault(session_id, Session())

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _retrieval_query(self, session: Session, question: str) -> str:
        """Fold the previous turn into the query so follow-ups retrieve well."""
        last_q = session.last_question()
        return f"{last_q} {question}".strip() if last_q else question

    def ask(self, question: str, session_id: str | None = None) -> ChatResponse:
        start = time.perf_counter()
        session_id = session_id or str(uuid.uuid4())
        session = self._get_session(session_id)

        if self.vectorstore.is_empty():
            answer = "No documents have been ingested yet. Upload a document first."
            session.add(question, answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                sources=[],
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        retrieval_query = self._retrieval_query(session, question)
        candidates: List[Document] = self.vectorstore.similarity_search(
            retrieval_query, k=settings.top_k_retrieve
        )
        ranked = self.reranker.rerank(retrieval_query, candidates, top_n=settings.top_n_context)

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}] {doc.page_content}"
            for doc, _ in ranked
        )
        prompt = QA_PROMPT.format(history=session.format(), context=context, question=question)
        answer = self.llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer = answer.content
        answer = str(answer).strip()

        session.add(question, answer)

        sources = [
            SourceSnippet(
                source=doc.metadata.get("source", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                score=float(score),
                text=doc.page_content[:300],
            )
            for doc, score in ranked
        ]

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=sources,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
