import os
from typing import Any, List, Optional

import numpy as np
from src.models import Chunk

EMBEDDINGS_PATH = "data/processed/embeddings.npy"
minilm = None


def build_MiniLM() -> Any:
    """Load the MiniLM sentence-embedding model once and reuse it."""
    global minilm
    if minilm is None:
        from sentence_transformers import SentenceTransformer
        minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return minilm


def build_embeddings(texts: List[str]) -> Any:
    """Encode texts into normalized embedding vectors."""
    return build_MiniLM().encode(texts, batch_size=64,
                                 show_progress_bar=True,
                                 normalize_embeddings=True)


def dump_embeddings(embeddings: Any) -> None:
    """Save embeddings to data/processed/embeddings.npy."""
    os.makedirs("data/processed", exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)


def load_embeddings(n_chunks: int) -> Optional[Any]:
    """Load saved embeddings, or None if missing or mismatched in size."""
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
    except Exception:
        return None
    if embeddings.shape[0] != n_chunks:
        return None
    return embeddings


def semantic_scores(query: str, embeddings: Any) -> Any:
    """Return similarity scores between the query and every chunk."""
    query_vector = build_MiniLM().encode([query],
                                         normalize_embeddings=True)[0]
    return embeddings @ query_vector


def update_embeddings(
    old_chunks: List[Chunk],
    kept: List[int],
    new_chunks: List[Chunk],
    chunks: List[Chunk],
) -> Any:
    """Reuse kept embeddings and encode only new chunks."""
    old_emb = None
    if old_chunks:
        old_emb = load_embeddings(len(old_chunks))
    if old_emb is None:
        texts = []
        for c in chunks:
            texts.append(c.text)
        return build_embeddings(texts)
    if not new_chunks:
        return old_emb[kept]
    texts = []
    for c in new_chunks:
        texts.append(c.text)
    return np.vstack([old_emb[kept], build_embeddings(texts)])
