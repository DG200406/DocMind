"""Loading and chunking of source documents into LangChain Document objects."""

import uuid
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docmind.config import settings

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def load_file(path: Path) -> List[Document]:
    """Load a single file into one or more Documents, tagged with source metadata."""
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": path.name})]

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_num},
                    )
                )
        return docs

    if suffix == ".docx":
        import docx

        document = docx.Document(str(path))
        text = "\n".join(p.text for p in document.paragraphs)
        return [Document(page_content=text, metadata={"source": path.name})]

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks, assigning each a stable chunk_id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
    return chunks


def load_and_chunk_file(path: Path) -> List[Document]:
    return chunk_documents(load_file(path))


def load_and_chunk_directory(directory: Path) -> List[Document]:
    all_chunks: List[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_chunks.extend(load_and_chunk_file(path))
    return all_chunks
