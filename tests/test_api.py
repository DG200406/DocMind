"""Smoke test for the FastAPI app. Requires the index to already be populated
(run `python scripts/ingest.py` first) since model loading is slow.

Usage:
    python -m pytest tests/test_api.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_chat_round_trip():
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "What is FAISS?"})
        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "session_id" in body
        assert isinstance(body["sources"], list)

        # Follow-up question reusing the same session should carry memory.
        follow_up = client.post(
            "/chat",
            json={"question": "What library is it from?", "session_id": body["session_id"]},
        )
        assert follow_up.status_code == 200
