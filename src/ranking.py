import numpy as np
from src.tokenizer import build_token
from src.embedding import semantic_scores

MODES = ("bm25", "semantic", "hybrid")


def get_ranks(scores):
    order = scores.argsort()[::-1]
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    return ranks


def rrf(bm25_scores, sem_scores):
    return 1 / (60 + get_ranks(bm25_scores)) + 1 / (60 + get_ranks(sem_scores))


def top_chunks(query, k, chunks, bm25, embeddings, mode):
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
