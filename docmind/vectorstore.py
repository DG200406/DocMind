"""FAISS-backed vector store over HuggingFace sentence-transformer embeddings."""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from docmind.config import settings


class VectorStoreManager:
    def __init__(self, index_dir: Optional[Path] = None, embedding_model: Optional[str] = None):
        self.index_dir = index_dir or settings.index_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model or settings.embedding_model
        )
        self.store: Optional[FAISS] = self._load()

    def _load(self) -> Optional[FAISS]:
        faiss_file = self.index_dir / "index.faiss"
        if faiss_file.exists():
            return FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    def save(self) -> None:
        if self.store is not None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.store.save_local(str(self.index_dir))

    def add_documents(self, chunks: List[Document]) -> int:
        if not chunks:
            return 0
        if self.store is None:
            self.store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.store.add_documents(chunks)
        self.save()
        return len(chunks)

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        if self.store is None:
            return []
        k = k or settings.top_k_retrieve
        return self.store.similarity_search(query, k=k)

    def list_sources(self) -> List[str]:
        if self.store is None:
            return []
        sources = {
            doc.metadata.get("source", "unknown")
            for doc in self.store.docstore._dict.values()
        }
        return sorted(sources)

    def chunk_count(self) -> int:
        if self.store is None:
            return 0
        return len(self.store.docstore._dict)

    def is_empty(self) -> bool:
        return self.store is None or self.chunk_count() == 0
