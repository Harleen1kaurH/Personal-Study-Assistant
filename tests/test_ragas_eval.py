"""
test_ragas_eval.py — Level 2 eval: LLM-judged RAGAS metrics.

MENTAL MODEL (continuing from test_assertions.py's Level 1):
  Level 1 (test_assertions.py) asks: "Is the pipeline functioning correctly?"
  Level 2 (this file)           asks: "Are the answers actually GOOD?"

  RAGAS metrics used here:
    Faithfulness        → Is the answer grounded in the retrieved context, or
                           did the LLM say things the context doesn't support?
    Answer Relevancy    → Does the answer actually address the question asked
                           (as opposed to being accurate but off-topic)?
    Context Precision   → Of the chunks retrieved, were the useful ones ranked
                           near the top? (needs the ground-truth answer)
    Context Recall      → Did retrieval pull in everything needed to produce
                           the ground-truth answer, or did it miss content?

  Faithfulness, Context Precision and Context Recall use an LLM-as-judge.
  Answer Relevancy additionally needs an embedding model (it re-generates a
  question from the answer and compares it to the original via cosine
  similarity) — we reuse the SAME embedding model the vectorstore uses
  (all-MiniLM-L6-v2), so relevancy scoring stays consistent with retrieval.

  The judge LLM is the project's own Groq model (see src/llm.py). That's a
  reasonable default for iterating locally; for a more trustworthy score a
  stronger/independent model (e.g. GPT-4o, Gemini 2.5 Pro) is normally used
  as judge so it isn't grading its own homework.

REQUIREMENTS:
  - synthetic_questions.json must exist (run tests/data/generate_inputs.py first)
  - Your document must already be embedded in ChromaDB
  - GROQ_API_KEY set in .env (same LLM the pipeline itself uses)

USAGE:
  pytest tests/test_ragas_eval.py -v -s          # run as a pass/fail gate
  python tests/test_ragas_eval.py                # run standalone, print + save a report
  python tests/test_ragas_eval.py --limit 5       # quick smoke run on first 5 questions
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

# ── Path setup ───────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import rag
from llm import get_llm
from vectorstore import list_collections, EMBEDDING_MODEL

from datasets import Dataset
from sentence_transformers import SentenceTransformer

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_PATH = Path(__file__).parent / "data" / "synthetic_questions.json"
RESULTS_CSV = Path(__file__).parent.parent / "eval_history.csv"

# Minimum acceptable mean score per metric (0.0-1.0). Tune these once you have
# a baseline run — they exist so a regression in retrieval or prompting fails
# the test suite instead of silently shipping.
THRESHOLDS = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "llm_context_precision_with_reference": 0.6,
    "context_recall": 0.6,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

class _SentenceTransformerEmbeddings:
    """
    Minimal langchain-style embeddings adapter (embed_query / embed_documents)
    around sentence-transformers, so RAGAS scores relevancy using the EXACT
    same embedding model the vectorstore uses for retrieval — no extra
    dependency needed beyond sentence-transformers, which the project already
    requires.
    """

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=False).tolist()


def load_dataset(limit: int | None = None) -> list[dict]:
    if not DATASET_PATH.exists():
        pytest.skip(
            f"{DATASET_PATH} not found. "
            "Run: python tests/data/generate_inputs.py <your_doc>"
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def get_file_hash() -> str:
    """Returns the first available ChromaDB collection (the embedded document)."""
    collections = list_collections()
    if not collections:
        pytest.skip(
            "No ChromaDB collections found. "
            "Run: python tests/data/generate_inputs.py <your_doc> to embed first."
        )
    return collections[0]


def run_pipeline(dataset: list[dict], file_hash: str) -> list[dict]:
    """Runs the full RAG pipeline once per question and shapes rows for RAGAS."""
    rows = []
    for item in dataset:
        result = rag.ask_question(item["question"], file_hash)
        rows.append({
            "user_input": result["question"],
            "response": result["answer"],
            "retrieved_contexts": [c["content"] for c in result["chunks"]],
            "reference": item["ground_truth"],
        })
    return rows


def run_ragas_eval(dataset: list[dict], file_hash: str):
    """Runs the pipeline + RAGAS scoring. Returns the ragas EvaluationResult."""
    rows = run_pipeline(dataset, file_hash)
    eval_dataset = Dataset.from_list(rows)

    # Reuse the project's own LLM (Groq) as the judge, and the same embedding
    # model used at retrieval time (all-MiniLM-L6-v2) for answer relevancy.
    judge_llm = LangchainLLMWrapper(get_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(
        _SentenceTransformerEmbeddings(EMBEDDING_MODEL)
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    return evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def file_hash():
    return get_file_hash()


@pytest.fixture(scope="session")
def ragas_scores(dataset, file_hash):
    """
    Runs the pipeline + RAGAS evaluation ONCE for the whole test session
    (LLM calls are slow/costly — don't repeat them per assertion), saves a
    per-question CSV for manual inspection, and hands back the mean scores.
    """
    result = run_ragas_eval(dataset, file_hash)
    df = result.to_pandas()
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[RAGAS] Per-question results saved to {RESULTS_CSV}")
    print(df.to_string())
    return df.mean(numeric_only=True)


# ── Threshold assertions ───────────────────────────────────────────────────────

class TestRagasQuality:
    """Are the answers actually good, according to RAGAS' LLM-judged metrics?"""

    def test_faithfulness(self, ragas_scores):
        score = ragas_scores["faithfulness"]
        assert score >= THRESHOLDS["faithfulness"], (
            f"Faithfulness {score:.3f} is below threshold "
            f"{THRESHOLDS['faithfulness']} — answers may be hallucinating "
            f"beyond the retrieved context."
        )

    def test_answer_relevancy(self, ragas_scores):
        score = ragas_scores["answer_relevancy"]
        assert score >= THRESHOLDS["answer_relevancy"], (
            f"Answer relevancy {score:.3f} is below threshold "
            f"{THRESHOLDS['answer_relevancy']} — answers may be drifting "
            f"off-topic from the question asked."
        )

    def test_context_precision(self, ragas_scores):
        score = ragas_scores["llm_context_precision_with_reference"]
        assert score >= THRESHOLDS["llm_context_precision_with_reference"], (
            f"Context precision {score:.3f} is below threshold "
            f"{THRESHOLDS['llm_context_precision_with_reference']} — "
            f"retrieved chunks may be poorly ranked (relevant chunks buried "
            f"below irrelevant ones). Consider tuning TOP_K or the embedding "
            f"model in src/vectorstore.py."
        )

    def test_context_recall(self, ragas_scores):
        score = ragas_scores["context_recall"]
        assert score >= THRESHOLDS["context_recall"], (
            f"Context recall {score:.3f} is below threshold "
            f"{THRESHOLDS['context_recall']} — retrieval may be missing "
            f"content needed to fully answer some questions. Consider "
            f"increasing TOP_K or reviewing chunking in src/loader.py."
        )


# ── Standalone runner ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation over tests/data/synthetic_questions.json"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N questions (useful for a quick smoke run)",
    )
    args = parser.parse_args()

    dataset = load_dataset(limit=args.limit)
    file_hash = get_file_hash()

    print(f"[RAGAS] Running pipeline on {len(dataset)} question(s) "
          f"(collection: {file_hash})...")
    result = run_ragas_eval(dataset, file_hash)
    df = result.to_pandas()

    print("\n=== Per-question scores ===")
    print(df.to_string())

    means = df.mean(numeric_only=True)
    print("\n=== Average scores ===")
    for metric, threshold in THRESHOLDS.items():
        score = means.get(metric, float("nan"))
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  {metric:45s} {score:.3f}  (threshold {threshold:.2f})  [{status}]")

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[RAGAS] Full results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
