"""Cross-encoder re-ranking of candidate chunks retrieved from FAISS."""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from docmind.config import settings


class Reranker:
    def __init__(self, model_name: Optional[str] = None):
        self.model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(
        self, query: str, candidates: List[Document], top_n: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []
        top_n = top_n or settings.top_n_context
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
