from rank_bm25 import BM25Okapi
from src.cache import get_index
from src.tokenizer import build_token


def build_bm25(tokenized):
    bm25 = BM25Okapi(tokenized)
    return bm25


def retrieve(query, k):
    if not query_tok:
        return []
    chunks, bm25 = get_index()
    query_tok = build_token(query)
    scores = bm25.get_scores(query_tok)
    if scores.max() <= 0:
        return []
    top_k_indices = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_k_indices]
