"""
============================================================
  MODULE 2 — processor/knowledge_store.py
  Responsibility: Clean → Chunk → Embed → Store in ChromaDB
============================================================

  Pipeline:
    raw docs → clean text → split into chunks
             → embed with SentenceTransformer
             → upsert into ChromaDB collection

  Also provides:  query(text, top_k) → relevant chunks
"""

import re
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
)


# ──────────────────────────────────────────
#  Singleton loader for embedding model
#  (loaded once, reused across calls)
# ──────────────────────────────────────────
_embedder = None

def get_embedder() -> SentenceTransformer:
    """
    Lazy-load the embedding model.

    Hyperparameter to tune:
      EMBEDDING_MODEL in settings.py
        "all-MiniLM-L6-v2"          → fastest, 384-dim
        "BAAI/bge-base-en-v1.5"     → better quality, 768-dim
        "BAAI/bge-small-en-v1.5"    → balanced
    """
    global _embedder
    if _embedder is None:
        print(f"  [Embedder] Loading model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


# ──────────────────────────────────────────
#  ChromaDB client
# ──────────────────────────────────────────
def get_collection():
    """
    Return (or create) the ChromaDB collection.
    Data persists in CHROMA_PERSIST_DIR across runs.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


# ──────────────────────────────────────────
#  Step 1 — Clean text
# ──────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Remove HTML tags, excess whitespace, and junk characters.
    Runs before chunking.
    """
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"http\S+", " ", text)            # remove URLs
    text = re.sub(r"[^\w\s.,!?;:()\-\"%$]", " ", text)  # keep readable chars
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


# ──────────────────────────────────────────
#  Step 2 — Chunk text
# ──────────────────────────────────────────
def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int    = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping windows.

    Hyperparameters to tune:
      CHUNK_SIZE    → bigger = more context per chunk, fewer chunks
      CHUNK_OVERLAP → bigger = less info loss at boundaries
    """
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap   # slide window
    return [c for c in chunks if len(c) > 50]   # drop tiny chunks


# ──────────────────────────────────────────
#  Step 3 — Build a stable document ID
# ──────────────────────────────────────────
def make_id(title: str, chunk_index: int) -> str:
    """SHA-256 based ID so re-runs don't duplicate data."""
    raw = f"{title}__chunk{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ──────────────────────────────────────────
#  Step 4 — Store documents in ChromaDB
# ──────────────────────────────────────────
def store_documents(documents: list[dict]) -> int:
    """
    Full pipeline: clean → chunk → embed → upsert.

    Args:
      documents: list from data_collector.collect_all()

    Returns:
      total number of chunks stored
    """
    collection = get_collection()
    embedder   = get_embedder()

    total_chunks = 0
    batch_ids, batch_docs, batch_metas, batch_embeddings = [], [], [], []
    BATCH_SIZE = 32   # process 32 chunks at a time (tune for RAM)

    print(f"\n  [Store] Processing {len(documents)} documents...")

    for doc in documents:
        clean = clean_text(doc["content"])
        chunks = chunk_text(clean)

        for i, chunk in enumerate(chunks):
            doc_id = make_id(doc["title"], i)

            # Skip if already in DB (idempotent)
            existing = collection.get(ids=[doc_id])
            if existing["ids"]:
                continue

            batch_ids.append(doc_id)
            batch_docs.append(chunk)
            batch_metas.append({
                "title":  doc["title"][:200],
                "source": doc["source"],
                "url":    doc["url"][:300],
                "date":   doc["date"][:20],
            })

            # Embed in batches for speed
            if len(batch_ids) >= BATCH_SIZE:
                embeddings = embedder.encode(batch_docs).tolist()
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=embeddings,
                )
                total_chunks += len(batch_ids)
                batch_ids, batch_docs, batch_metas = [], [], []

    # Flush remaining
    if batch_ids:
        embeddings = embedder.encode(batch_docs).tolist()
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )
        total_chunks += len(batch_ids)

    print(f"  [Store] ✅ {total_chunks} new chunks stored. Total in DB: {collection.count()}")
    return total_chunks


# ──────────────────────────────────────────
#  Query: semantic search
# ──────────────────────────────────────────
def query(text: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Find the most relevant chunks for a query.

    Hyperparameter to tune:
      TOP_K_RETRIEVAL → more = richer context for LLM, but slower

    Returns:
      list of {text, source, url, date, distance}
    """
    collection = get_collection()
    embedder   = get_embedder()

    embedding = embedder.encode([text]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", ""),
            "url":      meta.get("url", ""),
            "date":     meta.get("date", ""),
            "title":    meta.get("title", ""),
            "distance": round(dist, 4),
        })

    return chunks


# ──────────────────────────────────────────
#  Utility: collection stats
# ──────────────────────────────────────────
def get_stats() -> dict:
    """Return basic stats about the knowledge store."""
    collection = get_collection()
    count = collection.count()
    return {
        "total_chunks":    count,
        "collection_name": CHROMA_COLLECTION,
        "persist_dir":     CHROMA_PERSIST_DIR,
    }


if __name__ == "__main__":
    # Quick test
    results = query("NVIDIA AI GPU market")
    print(f"\nTop result:\n{results[0] if results else 'No results yet'}")
