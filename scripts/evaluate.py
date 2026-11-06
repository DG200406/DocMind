"""Measure retrieval quality with and without cross-encoder re-ranking.

Reproduces the "re-ranking raised Hit@1 from ~60% to ~80%" style metric:
for each test question, check whether the chunk from the expected source
document is ranked first, both using raw FAISS similarity order and after
re-ranking the top-K candidates with a cross-encoder.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --k 20 --questions tests/test_questions.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docmind.config import settings
from docmind.reranker import Reranker
from docmind.vectorstore import VectorStoreManager


def hit_at_1(source: str, expected: str) -> bool:
    return source == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=settings.top_k_retrieve)
    parser.add_argument("--questions", default="tests/test_questions.json")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    vectorstore = VectorStoreManager()
    if vectorstore.is_empty():
        print("Index is empty. Run `python scripts/ingest.py` first.")
        return

    reranker = Reranker()

    before_hits = 0
    after_hits = 0
    latencies = []

    for item in questions:
        question, expected = item["question"], item["expected_source"]

        start = time.perf_counter()
        candidates = vectorstore.similarity_search(question, k=args.k)
        latencies.append((time.perf_counter() - start) * 1000)

        if candidates and hit_at_1(candidates[0].metadata.get("source"), expected):
            before_hits += 1

        ranked = reranker.rerank(question, candidates, top_n=1)
        if ranked and hit_at_1(ranked[0][0].metadata.get("source"), expected):
            after_hits += 1

    n = len(questions)
    avg_latency = sum(latencies) / n if n else 0.0

    print(f"Questions evaluated: {n}")
    print(f"Hit@1 before re-ranking: {before_hits}/{n} ({100 * before_hits / n:.0f}%)")
    print(f"Hit@1 after re-ranking:  {after_hits}/{n} ({100 * after_hits / n:.0f}%)")
    print(f"Average FAISS search latency: {avg_latency:.1f} ms")


if __name__ == "__main__":
    main()
