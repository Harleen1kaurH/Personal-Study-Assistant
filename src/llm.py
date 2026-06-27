"""
llm.py — The single place where the LLM is configured.

MENTAL MODEL:
  LangChain's model wrappers give you a unified .invoke() interface.
  To swap from Gemini to GPT-4 or Claude: change TWO lines here.
  Nothing else in the codebase changes.

SWAP GUIDE:
  # Active (Gemini)
  from langchain_google_genai import ChatGoogleGenerativeAI
  llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

  # Swap to Claude:
  # from langchain_anthropic import ChatAnthropic
  # llm = ChatAnthropic(model="claude-opus-4-6")

  # Swap to OpenAI:
  # from langchain_openai import ChatOpenAI
  # llm = ChatOpenAI(model="gpt-4o")
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Returns the configured LLM instance.

    This is the ONLY function the rest of the codebase calls.
    Changing the model or provider: edit this function only.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Copy .env.example to .env and add your key from aistudio.google.com/apikey"
        )

    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",  # ← change this string to swap models
        google_api_key=api_key,
        temperature=0.2,  # Low temp = more focused, less creative. Good for factual Q&A.
    )
