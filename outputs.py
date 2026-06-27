"""
outputs.py — Pydantic schemas for structured outputs.

MENTAL MODEL:
  Prompt-based JSON ("return only JSON, no fences"):  ~85% reliable
  Pydantic + .with_structured_output():               ~99% reliable

  Why? The model is constrained to match your schema at the token level.
  You get back a typed Python object — access .cards, .questions directly.
  No json.loads(), no try/except, no stripping markdown fences.

SCHEMAS DEFINED:
  Flashcard       → front (question) + back (answer)
  FlashcardSet    → list of Flashcard
  QuizOption      → label ("A") + text (the option content)
  QuizQuestion    → question + 4 options + correct label + explanation
  Quiz            → list of QuizQuestion
"""

from pydantic import BaseModel, Field
from llm import get_llm


# ── Flashcard schemas ──────────────────────────────────────────────────────────

class Flashcard(BaseModel):
    front: str = Field(description="The question or concept prompt on the front of the card")
    back: str = Field(description="The concise answer or explanation on the back of the card")


class FlashcardSet(BaseModel):
    cards: list[Flashcard] = Field(description="List of flashcards generated from the content")


# ── Quiz schemas ───────────────────────────────────────────────────────────────

class QuizOption(BaseModel):
    label: str = Field(description="Option label: A, B, C, or D")
    text: str = Field(description="The text content of this answer option")


class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question")
    options: list[QuizOption] = Field(description="Four answer options labeled A through D")
    correct: str = Field(description="The label of the correct answer: A, B, C, or D")
    explanation: str = Field(description="Why the correct answer is right, grounded in the source material")


class Quiz(BaseModel):
    questions: list[QuizQuestion] = Field(description="List of quiz questions")


# ── Structured LLM factories ──────────────────────────────────────────────────
# These bind the schema to the LLM. Call them to get a schema-constrained LLM.
# Usage: structured_llm = get_flashcard_llm(); result = structured_llm.invoke(prompt)

def get_flashcard_llm():
    """Returns an LLM that always outputs a FlashcardSet — no parsing needed."""
    return get_llm().with_structured_output(FlashcardSet)


def get_quiz_llm():
    """Returns an LLM that always outputs a Quiz — no parsing needed."""
    return get_llm().with_structured_output(Quiz)
