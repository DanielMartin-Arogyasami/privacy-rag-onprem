"""Evaluate RAG pipeline on BioASQ benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from config.settings import get_settings
from src.models import QueryRequest
from src.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = [
    {"body": "What are the side effects of metformin?", "type": "summary"},
    {"body": "What is the mechanism of action of ibuprofen?", "type": "factoid"},
    {"body": "What drugs are used for drug-resistant tuberculosis?", "type": "list"},
    {"body": "How does CRISPR-Cas9 gene editing work?", "type": "summary"},
    {"body": "What are the known drug interactions of warfarin?", "type": "list"},
]


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        logger.info("BioASQ file not found — using sample questions")
        return SAMPLE_QUESTIONS
    with open(path) as f:
        data = json.load(f)
    return data.get("questions", data) if isinstance(data, dict) else data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bioasq_sample.json")
    parser.add_argument("--output", default="results/bioasq_results.json")
    args = parser.parse_args()

    pipeline = RAGPipeline(get_settings())
    questions = load_questions(Path(args.data))
    logger.info("Evaluating %d questions...", len(questions))

    results = {"total_questions": len(questions), "questions": []}
    totals = {"time": 0.0, "passages": 0, "confidence": 0.0, "abstentions": 0}

    for q in questions:
        start = time.time()
        resp = pipeline.query(QueryRequest(query=q["body"], user_id="eval"))
        elapsed = (time.time() - start) * 1000
        totals["time"] += elapsed
        totals["passages"] += len(resp.sources)
        totals["confidence"] += resp.confidence
        totals["abstentions"] += int(resp.is_abstention)
        results["questions"].append({"query": q["body"], "answer_preview": resp.answer[:300], "num_sources": len(resp.sources), "confidence": resp.confidence, "is_abstention": resp.is_abstention, "latency_ms": round(elapsed, 1)})

    n = max(len(questions), 1)
    results["avg_retrieval_time_ms"] = round(totals["time"] / n, 1)
    results["avg_confidence"] = round(totals["confidence"] / n, 3)
    results["abstention_rate"] = round(totals["abstentions"] / n, 3)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
