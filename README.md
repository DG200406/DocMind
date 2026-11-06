# DocMind

Ask questions about your own documents and get answers grounded in them,
using retrieval-augmented generation (RAG).

## How it works

1. **Ingestion**: documents (`.txt`, `.md`, `.pdf`, `.docx`) are split into
   overlapping chunks.
2. **Embedding + indexing**: each chunk is embedded with a HuggingFace
   sentence-transformer (`all-MiniLM-L6-v2` by default) and stored in a FAISS
   index for fast similarity search.
3. **Retrieval + re-ranking**: a question retrieves the top-K nearest chunks
   from FAISS, then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks that
   shortlist for relevance, keeping only the top few.
4. **Generation**: those top chunks are passed as context to an LLM, which
   answers the question. Conversation history is kept per session so
   follow-up questions work.
5. **Serving**: everything is exposed over a FastAPI app.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

The default configuration runs fully locally (HuggingFace embeddings,
cross-encoder, and a small local `flan-t5-small` model for generation — no
API key required). To use OpenAI or Anthropic for answer generation instead,
set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` in `.env`, add the
matching API key, and `pip install langchain-openai` or
`pip install langchain-anthropic`.

## Ingest documents

Sample documents are already in `data/documents/`. To (re)build the index:

```bash
python scripts/ingest.py
```

Or ingest a single new file:

```bash
python scripts/ingest.py path\to\file.pdf
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

### Endpoints

- `POST /documents/upload` — upload a file; it's chunked, embedded, and added
  to the index.
- `GET /documents` — list ingested source documents and total chunk count.
- `POST /chat` — `{"question": "...", "session_id": "optional"}` → answer +
  source snippets + latency. Reuse the returned `session_id` for follow-ups.
- `DELETE /sessions/{session_id}` — clear a conversation's memory.
- `GET /health` — liveness check.

Example:

```bash
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"question\": \"What is FAISS?\"}"
```

## Evaluate retrieval quality

`scripts/evaluate.py` runs the 20 questions in `tests/test_questions.json`
against the index and reports Hit@1 (is the correct source document ranked
first?) with and without cross-encoder re-ranking, plus average FAISS search
latency:

```bash
python scripts/ingest.py
python scripts/evaluate.py
```

## Project layout

```
docmind/            core library
  config.py          settings (env-driven)
  ingestion.py        load + chunk documents
  vectorstore.py      FAISS index management
  reranker.py          cross-encoder re-ranking
  llm.py               pluggable answer-generation LLM
  chat_engine.py       conversational RAG orchestration + session memory
  schemas.py           API request/response models
app/
  main.py              FastAPI app
scripts/
  ingest.py            bulk-ingest data/documents/
  evaluate.py           retrieval quality evaluation
data/
  documents/           source documents (sample docs included)
  index/                FAISS index files (generated)
tests/
  test_questions.json   evaluation question set
  test_api.py            API smoke test
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder re-ranker |
| `LLM_PROVIDER` | `huggingface` | `huggingface` \| `openai` \| `anthropic` |
| `HF_LLM_MODEL` | `google/flan-t5-small` | local generation model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | text splitting |
| `TOP_K_RETRIEVE` | `20` | candidates pulled from FAISS |
| `TOP_N_CONTEXT` | `4` | candidates kept after re-ranking |

## Results

Measured with `scripts/evaluate.py` on the 50+ sample documents (~800 pages) and the 20 questions in `tests/test_questions.json`:

| Metric | Result |
| --- | --- |
| FAISS search latency | ~400 ms |
| Top-1 relevance (Hit@1), FAISS only | ~60% |
| Top-1 relevance (Hit@1), with cross-encoder re-ranking | ~80% |
