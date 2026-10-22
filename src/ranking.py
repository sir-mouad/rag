from typing import Any, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from src.models import Chunk
from src.tokenizer import build_token
from src.embedding import semantic_scores

MODES = ("bm25", "semantic", "hybrid")


def get_ranks(scores: Any) -> Any:
    """Return the rank of each score, where the highest score has rank 0."""
    order = scores.argsort()[::-1]
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    return ranks


def rrf(bm25_scores: Any, sem_scores: Any) -> Any:
    """Combine two score arrays with reciprocal rank fusion."""
    return 1 / (60 + get_ranks(bm25_scores)) + 1 / (60 + get_ranks(sem_scores))


def top_chunks(
    query: str,
    k: int,
    chunks: List[Chunk],
    bm25: BM25Okapi,
    embeddings: Optional[Any],
    mode: str,
) -> List[Chunk]:
    """Return the top-k chunks for a query in the chosen mode."""
    if mode not in MODES:
        raise ValueError("mode must be bm25, semantic or hybrid")
    query_tok = build_token(query)
    if not query_tok:
        return []
    bm25_scores = bm25.get_scores(query_tok)
    if mode == "bm25":
        if bm25_scores.max() <= 0:
            return []
        scores = bm25_scores
    else:
        sem_scores = semantic_scores(query, embeddings)
        if mode == "semantic" or bm25_scores.max() <= 0:
            scores = sem_scores
        else:
            scores = rrf(bm25_scores, sem_scores)
    top_k_indices = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_k_indices]
