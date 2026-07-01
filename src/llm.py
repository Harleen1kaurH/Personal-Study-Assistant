"""
llm.py — The single place where the LLM is configured.

MENTAL MODEL:
  LangChain's model wrappers give you a unified .invoke() interface.
  To swap from Gemini to GPT-4 or Claude: change TWO lines here.
  Nothing else in the codebase changes.

SWAP GUIDE:
  # Active (Groq — temporary for test phase)
  from langchain_groq import ChatGroq
  llm = ChatGroq(model="llama-3.3-70b-versatile")

  # Swap back to Gemini (production):
  # from langchain_google_genai import ChatGoogleGenerativeAI
  # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

  # Swap to OpenAI:
  # from langchain_openai import ChatOpenAI
  # llm = ChatOpenAI(model="gpt-4o-mini")
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm() -> ChatGroq:
    """
    Returns the configured LLM instance.

    This is the ONLY function the rest of the codebase calls.
    Changing the model or provider: edit this function only.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add your key to .env — get a free key at console.groq.com"
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",  # ← change this string to swap models
        api_key=api_key,
        temperature=0.2,  # Low temp = more focused, less creative. Good for factual Q&A.
    )
