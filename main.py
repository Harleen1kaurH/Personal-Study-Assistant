"""
main.py — Terminal testing harness.

Run this to test the full pipeline end-to-end before touching the UI.

Usage:
  python main.py path/to/your/notes.pdf
  python main.py path/to/your/notes.md

What it does:
  1. Loads and chunks the document
  2. Embeds chunks (or loads from disk if already done)
  3. Runs 3 demo modes: Q&A → Flashcards → Quiz
  4. Prints everything to terminal for inspection

Use this to:
  - Verify chunking looks sensible
  - Check retrieval is returning relevant chunks
  - Confirm flashcards/quiz are grounded in your content
  - Debug before building the Streamlit UI
"""

import sys
import os

# Add src/ to path so imports work from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from loader import load_and_chunk
from vectorstore import embed_and_store, collection_exists
import rag


def run_pipeline(doc_path: str):
    print("=" * 60)
    print("PERSONAL STUDY ASSISTANT — Pipeline Test")
    print("=" * 60)

    # ── Step 1: Load and chunk ─────────────────────────────────────
    print("\n[1/4] Loading and chunking document...")
    chunks, file_hash = load_and_chunk(doc_path)
    print(f"      Hash: {file_hash[:16]}...")
    print(f"      Chunks: {len(chunks)}")

    # Print first chunk as a sanity check
    if chunks:
        print(f"\n  First chunk preview:")
        # print(f"  Section: {chunks[0]['metadata']['section'] or '(none)'}")
        print(f"  Content: {chunks[0]['content'][:200]}...")

    # ── Step 2: Embed and store ────────────────────────────────────
    print("\n[2/4] Embedding...")
    if collection_exists(file_hash):
        print("      Already embedded — loading from disk instantly.")
    embed_and_store(chunks, file_hash)

    # ── Step 3: Q&A ───────────────────────────────────────────────
    print("\n[3/4] Q&A Test")
    print("-" * 40)

    # You can change this question to anything relevant to your doc
    question = "Whose resume is this? Give one line answer?"
    result = rag.ask_question(question, file_hash)

    print(f"\nQ: {result['question']}")
    print(f"\nA: {result['answer'][0]['text']}")

    # print(f"\nRetrieved {len(result['chunks'])} chunks:")
    
    for i, chunk in enumerate(result["chunks"], 1):
        section = chunk["metadata"].get("section", "(no section)")
        sim = chunk["similarity"]
        preview = chunk["content"][:80].replace("\n", " ")
        print(f"  [{i}] sim={sim} | {section} | {preview}...")

    # ── Step 4: Flashcards ────────────────────────────────────────
    print("\n[4a/4] Flashcard Generation")
    print("-" * 40)
    flashcard_set = rag.generate_flashcards(file_hash, num_cards=3)
    for i, card in enumerate(flashcard_set.cards, 1):
        print(f"\n  Card {i}:")
        print(f"    FRONT: {card.front}")
        print(f"    BACK:  {card.back}")

    # ── Step 4b: Quiz ─────────────────────────────────────────────
    print("\n[4b/4] Quiz Generation")
    print("-" * 40)
    quiz = rag.generate_quiz(file_hash, num_questions=2)
    for i, q in enumerate(quiz.questions, 1):
        print(f"\n  Q{i}: {q.question}")
        for opt in q.options:
            marker = "✓" if opt.label == q.correct else " "
            print(f"    [{marker}] {opt.label}. {opt.text}")
        print(f"    Explanation: {q.explanation}")

    print("\n" + "=" * 60)
    print("Pipeline complete. Run `streamlit run src/app.py` for the UI.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf_or_md>")
        print("Example: python main.py data/my_notes.pdf")
        sys.exit(1)

    doc_path = sys.argv[1]
    if not os.path.exists(doc_path):
        print(f"File not found: {doc_path}")
        sys.exit(1)

    run_pipeline(doc_path)
