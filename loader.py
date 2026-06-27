"""
loader.py — Document loading and smart chunking.

MENTAL MODEL:
  Character splitting is BLIND — it cuts mid-sentence, merges unrelated ideas.
  Structure splitting is AWARE — it respects the boundaries the author put there.

  smart_split() strategy:
    1. Try to split by Markdown headers (h1 > h2 > h3)
    2. For any section that's still too large → fall back to RecursiveCharacterTextSplitter
    3. Always return plain list[dict] — no LangChain objects escape this file

  Why convert to plain dicts?
    Nothing downstream should know or care about LangChain's Document class.
    Plain dicts = portable, testable, framework-agnostic.

CHUNK FORMAT:
  {
    "content":  "the actual text of this chunk",
    "metadata": {
        "source":   "data/my_notes.pdf",
        "section":  "Neural Networks > Backpropagation",  ← for citation UX
        "chunk_id": 0
    }
  }
"""

import hashlib
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800        # characters per chunk — sweet spot for RAG retrieval
CHUNK_OVERLAP = 100     # overlap so context doesn't get cut at boundaries


# ── PDF extraction ─────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """
    Extracts all text from a PDF file.

    Returns raw text as a single string.
    Use smart_split() on the result to turn it into chunks.
    """
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())

    full_text = "\n\n".join(pages)
    print(f"  Loaded {len(reader.pages)} pages from {Path(path).name} ({len(full_text):,} chars)")
    return full_text


def load_markdown(path: str) -> str:
    """Reads a .md file and returns its text content."""
    text = Path(path).read_text(encoding="utf-8")
    print(f"  Loaded {Path(path).name} ({len(text):,} chars)")
    return text


def load_document(path: str) -> str:
    """
    Auto-detects file type and loads accordingly.
    Supports .pdf and .md files.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".md", ".txt"):
        return load_markdown(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .md, .txt")


# ── Chunking ───────────────────────────────────────────────────────────────────

def _build_section_label(metadata: dict) -> str:
    """
    Converts LangChain header metadata into a readable section path.

    Example: {"Header 1": "Neural Networks", "Header 2": "Backprop"}
             → "Neural Networks > Backprop"
    """
    parts = []
    for key in ["Header 1", "Header 2", "Header 3"]:
        if key in metadata and metadata[key]:
            parts.append(metadata[key])
    return " > ".join(parts) if parts else ""


def smart_split(text: str, source: str) -> list[dict]:
    """
    Chunks text using a structure-first strategy with a size safety net.

    Strategy:
      1. Split by Markdown headers (respects author's own structure)
      2. For any chunk > CHUNK_SIZE → split further by characters
      3. Convert all results to plain dicts immediately

    Args:
      text:   Raw text content (from load_document)
      source: File path string — stored in metadata for citation

    Returns:
      list[dict] with "content" and "metadata" keys
      GUARANTEE: No LangChain Document objects returned.
    """
    # Step 1: Try structure splitting via Markdown headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        strip_headers=False,  # Keep the header text in the chunk for context
    )

    # Step 2: Size safety net — handle oversized sections
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraph breaks first
    )

    # Run structure split
    header_chunks = header_splitter.split_text(text)

    # Convert to plain dicts — LangChain objects stop here
    final_chunks = []
    chunk_id = 0

    for doc in header_chunks:
        section_label = _build_section_label(doc.metadata)
        content = doc.page_content.strip()

        if not content:
            continue

        # If the section is too large, split it further by characters
        if len(content) > CHUNK_SIZE:
            sub_texts = char_splitter.split_text(content)
            for sub_text in sub_texts:
                if sub_text.strip():
                    final_chunks.append({
                        "content": sub_text.strip(),
                        "metadata": {
                            "source": source,
                            "section": section_label,
                            "chunk_id": chunk_id,
                        }
                    })
                    chunk_id += 1
        else:
            final_chunks.append({
                "content": content,
                "metadata": {
                    "source": source,
                    "section": section_label,
                    "chunk_id": chunk_id,
                }
            })
            chunk_id += 1

    print(f"  Split into {len(final_chunks)} chunks")
    return final_chunks


# ── File ID helper ─────────────────────────────────────────────────────────────

def get_file_hash(path: str) -> str:
    """
    Returns an MD5 hash of the file's contents.

    Used by vectorstore.py to check if a file has already been embedded.
    Same file = same hash = skip re-embedding.
    """
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_and_chunk(path: str) -> tuple[list[dict], str]:
    """
    Full pipeline: load file → extract text → smart_split into chunks.

    Returns:
      (chunks, file_hash)
      chunks:    list[dict] ready for embedding
      file_hash: MD5 string used as ChromaDB collection name
    """
    print(f"\nLoading: {path}")
    text = load_document(path)
    file_hash = get_file_hash(path)
    chunks = smart_split(text, source=path)
    return chunks, file_hash
