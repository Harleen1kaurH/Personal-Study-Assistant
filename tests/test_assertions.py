"""
test_assertions.py — Level 1 eval: fast, deterministic pytest assertions.

MENTAL MODEL (from Hamel Husain's eval framework):
  Level 1 tests are NOT about whether the answer is "good".
  They are about whether the pipeline is functioning correctly.
  No LLM calls here — just code checking code.

  Think of these as: "If any of these fail, something is broken."

  Three categories:
    Structural  → is the answer well-formed?
    Retrieval   → did ChromaDB actually do its job?
    Safety      → did internal data leak into the response?

REQUIREMENTS:
  - synthetic_questions.json must exist (run generate_inputs.py first)
  - Your document must already be embedded in ChromaDB (run generate_inputs.py)
  - Set GOOGLE_API_KEY in your .env file

USAGE:
  pytest tests/ -v
  pytest tests/test_assertions.py -v -k "structural"   # run one category
  pytest tests/test_assertions.py -v --tb=short        # compact output
"""

import json
import re
import sys
import os
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import rag
from vectorstore import list_collections

# ── Load dataset ───────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data/synthetic_questions.json")
TOP_K = 4  # Must match vectorstore.TOP_K


def load_dataset() -> list[dict]:
    if not os.path.exists(DATASET_PATH):
        pytest.skip(
            "synthetic_questions.json not found. "
            "Run: python tests/data/generate_inputs.py <your_doc>"
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_hash() -> str:
    """Returns the first available ChromaDB collection (the embedded document)."""
    collections = list_collections()
    if not collections:
        pytest.skip(
            "No ChromaDB collections found. "
            "Run: python tests/data/generate_inputs.py <your_doc> to embed first."
        )
    return collections[0]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def file_hash():
    return get_file_hash()


@pytest.fixture(scope="session")
def pipeline_results(dataset, file_hash):
    """
    Runs the full RAG pipeline on every question in the dataset once.
    Cached for the whole test session — no repeated API calls.
    """
    results = []
    for item in dataset:
        result = rag.ask_question(item["question"], file_hash)
        results.append(result)
    return results


# ── Structural assertions ──────────────────────────────────────────────────────

class TestStructural:
    """Does the pipeline return well-formed answers?"""

    def test_answer_is_not_none(self, pipeline_results):
        for result in pipeline_results:
            assert result["answer"] is not None, (
                f"Got None answer for question: {result['question']}"
            )

    def test_answer_is_string(self, pipeline_results):
        for result in pipeline_results:
            assert isinstance(result["answer"], str), (
                f"Answer is {type(result['answer'])}, expected str. "
                f"Question: {result['question']}"
            )

    def test_answer_not_empty(self, pipeline_results):
        for result in pipeline_results:
            assert result["answer"].strip() != "", (
                f"Got empty answer for question: {result['question']}"
            )

    def test_answer_not_too_short(self, pipeline_results):
        """Answers under 5 words are suspiciously short — likely a failure."""
        MIN_WORDS = 5
        for result in pipeline_results:
            word_count = len(result["answer"].split())
            assert word_count >= MIN_WORDS, (
                f"Answer has only {word_count} words (min: {MIN_WORDS}). "
                f"Question: {result['question']}\n"
                f"Answer: {result['answer']}"
            )

    def test_answer_not_too_long(self, pipeline_results):
        """Answers over 400 words suggest the LLM is rambling or ignoring the prompt."""
        MAX_WORDS = 400
        for result in pipeline_results:
            word_count = len(result["answer"].split())
            assert word_count <= MAX_WORDS, (
                f"Answer has {word_count} words (max: {MAX_WORDS}). "
                f"Question: {result['question']}"
            )

    def test_question_echoed_correctly(self, pipeline_results, dataset):
        """The pipeline must return the original question unchanged."""
        for result, item in zip(pipeline_results, dataset):
            assert result["question"] == item["question"], (
                f"Question mismatch. Expected: {item['question']!r}, "
                f"Got: {result['question']!r}"
            )


# ── Retrieval assertions ───────────────────────────────────────────────────────

class TestRetrieval:
    """Did ChromaDB retrieve chunks as expected?"""

    def test_chunks_not_empty(self, pipeline_results):
        """At least one chunk must be retrieved for every question."""
        for result in pipeline_results:
            assert len(result["chunks"]) > 0, (
                f"No chunks retrieved for question: {result['question']}"
            )

    def test_chunks_count_matches_top_k(self, pipeline_results):
        """
        Retrieved chunk count should equal TOP_K (unless the collection
        has fewer chunks than TOP_K — that's a small-doc edge case).
        """
        for result in pipeline_results:
            count = len(result["chunks"])
            assert count <= TOP_K, (
                f"Retrieved {count} chunks but TOP_K is {TOP_K}. "
                f"Question: {result['question']}"
            )

    def test_chunks_have_content(self, pipeline_results):
        """Every retrieved chunk must have non-empty content."""
        for result in pipeline_results:
            for i, chunk in enumerate(result["chunks"]):
                assert "content" in chunk, (
                    f"Chunk {i} missing 'content' key. Question: {result['question']}"
                )
                assert chunk["content"].strip() != "", (
                    f"Chunk {i} has empty content. Question: {result['question']}"
                )

    def test_chunks_have_similarity_score(self, pipeline_results):
        """Every chunk must have a similarity score (0.0 to 1.0)."""
        for result in pipeline_results:
            for i, chunk in enumerate(result["chunks"]):
                assert "similarity" in chunk, (
                    f"Chunk {i} missing 'similarity' key. Question: {result['question']}"
                )
                sim = chunk["similarity"]
                assert 0.0 <= sim <= 1.0, (
                    f"Similarity {sim} out of range [0, 1]. "
                    f"Chunk {i}, Question: {result['question']}"
                )


# ── Safety assertions ──────────────────────────────────────────────────────────

class TestSafety:
    """Did internal data leak into user-facing answers?"""

    # UUID pattern: 8-4-4-4-12 hex chars
    UUID_PATTERN = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    )

    # MD5 hash pattern: 32 hex chars (ChromaDB collection names)
    MD5_PATTERN = re.compile(r"\b[0-9a-f]{32}\b", re.IGNORECASE)

    def test_no_uuid_in_answer(self, pipeline_results):
        """ChromaDB document IDs (UUIDs) must never surface in answers."""
        for result in pipeline_results:
            match = self.UUID_PATTERN.search(result["answer"])
            assert match is None, (
                f"UUID found in answer: {match.group()!r}. "
                f"Question: {result['question']}"
            )

    def test_no_md5_hash_in_answer(self, pipeline_results):
        """ChromaDB collection names (MD5 hashes) must never appear in answers."""
        for result in pipeline_results:
            match = self.MD5_PATTERN.search(result["answer"])
            assert match is None, (
                f"MD5 hash found in answer: {match.group()!r}. "
                f"Question: {result['question']}"
            )

    def test_no_chroma_metadata_keywords(self, pipeline_results):
        """
        Internal ChromaDB field names should never appear in user-facing answers.
        If they do, internal metadata is leaking through the prompt or context.
        """
        internal_keywords = ["chunk_id", "hnsw:space", "collection_name"]
        for result in pipeline_results:
            answer_lower = result["answer"].lower()
            for keyword in internal_keywords:
                assert keyword not in answer_lower, (
                    f"Internal keyword {keyword!r} found in answer. "
                    f"Question: {result['question']}"
                )
