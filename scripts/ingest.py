"""Bulk-ingest every supported file in DOCS_DIR into the FAISS index.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py path/to/a/single/file.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docmind.config import settings
from docmind.ingestion import load_and_chunk_directory, load_and_chunk_file
from docmind.vectorstore import VectorStoreManager


def main():
    vectorstore = VectorStoreManager()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        chunks = load_and_chunk_file(path)
    else:
        path = settings.docs_path
        chunks = load_and_chunk_directory(path)

    added = vectorstore.add_documents(chunks)
    print(f"Ingested {added} chunks from {path}")
    print(f"Index now has {vectorstore.chunk_count()} chunks across {len(vectorstore.list_sources())} documents")


if __name__ == "__main__":
    main()
