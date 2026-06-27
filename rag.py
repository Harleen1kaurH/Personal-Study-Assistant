"""
rag.py — The full RAG pipeline.

MENTAL MODEL:
  Semantic search  = retrieval only (no LLM involved)
  RAG              = semantic search  +  generation
  RAG Agent        = LLM IN the retrieval loop (Phase 2)

  This file = standard RAG. Fixed pipeline. You control every step.

  The flow:
    question
      → embed question
      → cosine similarity vs stored chunks
      → top 4 chunks retrieved
      → build grounded prompt (context + question)
      → LLM generates answer
      → answer is grounded in YOUR documents

WHY CHUNK METADATA IN THE PROMPT?
  We pass section headings alongside content.
  Gemini can then say "According to Neural Networks > Backpropagation..."
  If it cites a section that doesn't exist → hallucination detected.
  This is a simple but effective honesty mechanism.

FLASHCARDS AND QUIZZES:
  Same retrieval → different prompt → schema-constrained output.
  ask_question()     → plain text answer
  generate_flashcards() → FlashcardSet (typed Pydantic object)
  generate_quiz()       → Quiz (typed Pydantic object)
"""

from llm import get_llm
from outputs import get_flashcard_llm, get_quiz_llm, FlashcardSet, Quiz
import vectorstore


# ── Prompt builders ────────────────────────────────────────────────────────────

def _format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into the context block injected into the prompt.

    Includes section metadata so the LLM can cite its sources.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk["metadata"].get("section", "")
        source = chunk["metadata"].get("source", "")
        header = f"[Chunk {i}"
        if section:
            header += f" | Section: {section}"
        if source:
            header += f" | Source: {source}"
        header += "]"
        parts.append(f"{header}\n{chunk['content']}")

    return "\n\n---\n\n".join(parts)


def _build_qa_prompt(question: str, context: str) -> str:
    return f"""You are a study assistant. Answer the question using ONLY the provided context from the student's notes.

Rules:
- If the context doesn't contain enough information, say so clearly. Don't hallucinate.
- Cite which section your answer comes from (e.g., "According to [Section: X]... but if section is not given then there is no need to write").
- Be concise but complete. Use bullet points for multi-part answers.

CONTEXT FROM NOTES:
{context}

QUESTION: {question}

ANSWER:"""


def _build_flashcard_prompt(context: str, num_cards: int) -> str:
    return f"""You are a study assistant creating flashcards from a student's notes.

Generate exactly {num_cards} flashcards from the content below.

Rules:
- Each card's FRONT should be a question or concept prompt (not trivia — test real understanding).
- Each card's BACK should be a concise, accurate answer grounded in the text.
- Cover the most important concepts in the content.
- Vary question types: definition, explain-why, compare, application.

CONTENT:
{context}"""


def _build_quiz_prompt(context: str, num_questions: int) -> str:
    return f"""You are a study assistant creating a multiple-choice quiz from a student's notes.

Generate exactly {num_questions} quiz questions from the content below.

Rules:
- Each question must have exactly 4 options labeled A, B, C, D.
- Only one option should be correct.
- The explanation must cite why the correct answer is right, grounded in the text.
- Make distractors plausible — don't make wrong answers obviously wrong.
- Test understanding, not memorization of exact phrases.

CONTENT:
{context}"""


# ── Core RAG functions ─────────────────────────────────────────────────────────

def ask_question(question: str, file_hash: str) -> dict:
    """
    Full RAG pipeline: question → retrieve → generate → answer.

    Args:
      question:  The student's question (plain text)
      file_hash: Which document to search (from loader.get_file_hash())

    Returns:
      {
        "answer":  str,          ← LLM's grounded answer
        "chunks":  list[dict],   ← retrieved chunks (for transparency/debugging)
        "question": str
      }
    """
    print(f"\n[RAG] Question: {question}")

    # Step 1: Semantic retrieval
    print(f"[RAG] Retrieving top chunks...")
    chunks = vectorstore.retrieve(query=question, file_hash=file_hash)
    print(f"[RAG] Retrieved {len(chunks)} chunks (similarities: {[c['similarity'] for c in chunks]})")

    # Step 2: Build grounded prompt
    context = _format_context(chunks)
    prompt = _build_qa_prompt(question, context)

    # Step 3: Generate answer
    print("[RAG] Generating answer...")
    llm = get_llm()
    response = llm.invoke(prompt)
    answer = response.content

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
    }


def generate_flashcards(file_hash: str, num_cards: int = 5) -> FlashcardSet:
    """
    Retrieves broad context from the document and generates flashcards.

    Uses a generic "key concepts" query to pull a representative sample.
    Returns a FlashcardSet (typed Pydantic object — access .cards directly).

    Args:
      file_hash: Which document to use
      num_cards: How many flashcards to generate (default: 5)

    Returns:
      FlashcardSet with .cards = list[Flashcard], each has .front and .back
    """
    print(f"\n[Flashcards] Generating {num_cards} cards...")

    # Use a broad retrieval query to get representative content
    chunks = vectorstore.retrieve(
        query="main concepts definitions key ideas important topics",
        file_hash=file_hash,
        top_k=6,  # Grab more chunks for flashcard generation
    )

    context = _format_context(chunks)
    prompt = _build_flashcard_prompt(context, num_cards)

    structured_llm = get_flashcard_llm()
    result: FlashcardSet = structured_llm.invoke(prompt)

    print(f"[Flashcards] Generated {len(result.cards)} cards")
    return result


def generate_quiz(file_hash: str, num_questions: int = 3) -> Quiz:
    """
    Retrieves broad context from the document and generates a quiz.

    Returns a Quiz (typed Pydantic object — access .questions directly).

    Args:
      file_hash:     Which document to use
      num_questions: How many questions to generate (default: 3)

    Returns:
      Quiz with .questions = list[QuizQuestion]
      Each QuizQuestion has: .question, .options, .correct, .explanation
    """
    print(f"\n[Quiz] Generating {num_questions} questions...")

    chunks = vectorstore.retrieve(
        query="important concepts principles methods examples",
        file_hash=file_hash,
        top_k=6,
    )

    context = _format_context(chunks)
    prompt = _build_quiz_prompt(context, num_questions)

    structured_llm = get_quiz_llm()
    result: Quiz = structured_llm.invoke(prompt)

    print(f"[Quiz] Generated {len(result.questions)} questions")
    return result


def ask_question_with_topic(question: str, file_hash: str, topic_hint: str = "") -> dict:
    """
    Variant of ask_question() where you can provide an extra topic hint
    to steer retrieval. Useful when the question is short/vague.

    Example:
      ask_question_with_topic(
          question="How does this work?",
          file_hash=hash,
          topic_hint="transformer attention mechanism"
      )
    """
    combined_query = f"{question} {topic_hint}".strip()
    chunks = vectorstore.retrieve(query=combined_query, file_hash=file_hash)
    context = _format_context(chunks)
    prompt = _build_qa_prompt(question, context)

    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "chunks": chunks,
    }
