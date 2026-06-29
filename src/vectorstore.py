"""
vectorstore.py — Embedding, persistence, and semantic retrieval.

MENTAL MODEL:
  Embedding model = a transformer pointed at MEANING COMPRESSION.
  It turns text → a dense vector (list of floats) where similar meaning = similar direction.

  Semantic search = "find vectors closest to the query vector" (cosine similarity).
  No keyword matching. "car" and "automobile" will retrieve each other.

  ChromaDB.Client()           → in-memory, dies when script ends
  ChromaDB.PersistentClient() → writes to disk, lives forever

HASH CHECK PATTERN:
  MD5 hash of file contents → used as ChromaDB collection name.
  On startup: collection exists? → skip embedding (instant load).
              collection missing? → embed all chunks and store.

  Same PDF, second run = zero embedding work.

EMBEDDING MODEL:
  EMBEDDING_MODEL is a single constant. To upgrade: change one string.

  Current:    all-MiniLM-L6-v2  (56.3 MTEB — fast, free, good enough)
  Upgrade 1:  BAAI/bge-small-en-v1.5  (62.2 MTEB)
  Upgrade 2:  intfloat/e5-large-v2    (64.9 MTEB)

RETRIEVAL:
  Top-k cosine similarity. Returns the 4 most relevant chunks to a query.
  Those chunks become the context injected into the RAG prompt.
"""

import chromadb
from chromadb.utils import embedding_functions

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"   # Persistent storage directory
TOP_K = 4                     # Number of chunks to retrieve per query
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ← change this string to upgrade

# ── Embedding function ─────────────────────────────────────────────────────────
# ChromaDB has a built-in wrapper for sentence-transformers.
# This runs locally — no API calls, no cost, no rate limits.

# An embedding model is the underlying machine learning algorithm (like BERT or
# OpenAI's text-embedding-3-small) that translates raw data into mathematical vectors, capturing 
# semantic meaning. An embedding function is the practical, programmatic wrapper that applies this model,
# handling the pipeline of tokenizing inputs and retrieving the final numerical vector.

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)


def _get_client() -> chromadb.PersistentClient:
    """Returns a ChromaDB client that persists to disk. Like a database connection, but for embeddings."""
    return chromadb.PersistentClient(path=CHROMA_PATH)



def collection_exists(file_hash: str) -> bool:
    """
    Checks whether a collection (= embedded file) already exists on disk.

    Args:
      file_hash: MD5 hash returned by loader.get_file_hash()

    Returns:
      True if already embedded, False if embedding needed.
    """
    client = _get_client()
    existing = [col.name for col in client.list_collections()]
    return file_hash in existing


def embed_and_store(chunks: list[dict], file_hash: str) -> None:
    """
    Embeds chunks and stores them in a persistent ChromaDB collection.

    SKIP LOGIC: If collection already exists (same file hash), does nothing.

    Args:
      chunks:    list[dict] from loader.smart_split() — has "content" and "metadata"
      file_hash: MD5 hash of the source file — used as the collection name
    """
    client = _get_client()

    # Hash check — skip if already embedded
    if collection_exists(file_hash):
        print(f"  Collection '{file_hash[:8]}...' already exists. Skipping embedding.")
        return

    print(f"  Embedding {len(chunks)} chunks into collection '{file_hash[:8]}...'")

    collection = client.create_collection(
        name=file_hash,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity for retrieval
    )

    # ChromaDB expects lists of: documents, metadatas, ids
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"{file_hash}_{chunk['metadata']['chunk_id']}" for chunk in chunks]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"  Stored {len(chunks)} chunks. Embeddings persisted to {CHROMA_PATH}/")


def retrieve(query: str, file_hash: str, top_k: int = TOP_K) -> list[dict]:
    """
    Finds the top_k most semantically similar chunks to a query.

    HOW IT WORKS:
      1. Embeds the query using the same model used at store time
      2. Computes cosine similarity against all stored vectors
      3. Returns the top_k closest chunks with their content and metadata

    Args:
      query:     The user's question (plain text)
      file_hash: Which collection to search (which document)
      top_k:     How many chunks to return (default: 4)

    Returns:
      list[dict] — each dict has "content" and "metadata" keys
      Same format as loader.smart_split() output.
    """
    client = _get_client()
    collection = client.get_collection(
        name=file_hash,
        embedding_function=_embedding_fn,
    )

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),  # Can't request more than we have
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested response format
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "content": doc,
            "metadata": meta,
            "similarity": round(1 - dist, 3),  # ChromaDB returns distance; convert to similarity
        })

    return chunks


def list_collections() -> list[str]:
    """Returns all stored collection names (file hashes). Useful for debugging."""
    client = _get_client()
    return [col.name for col in client.list_collections()]


def delete_collection(file_hash: str) -> None:
    """
    Removes a collection from disk. Use when you want to re-embed a file.
    Normally not needed — the hash check handles freshness.
    """
    client = _get_client()
    client.delete_collection(name=file_hash)
    print(f"  Deleted collection '{file_hash[:8]}...'")
