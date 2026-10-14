from src.cache import get_index, load_queries, dump_queries
from src.embedding import load_embeddings
from src.ranking import top_chunks, MODES


def get_embeddings(chunks, mode):
    if mode not in MODES:
        raise ValueError("mode must be bm25, semantic or hybrid")
    if mode == "bm25":
        return None
    embeddings = load_embeddings(len(chunks))
    if embeddings is None:
        raise ValueError("Embeddings not found: "
                         "run index with --semantic True")
    return embeddings


def retrieve(query, k, mode="bm25"):
    key = f"{mode}|{k}|{query.strip()}"
    entries = load_queries()
    if key in entries:
        return entries[key]
    chunks, bm25 = get_index()
    embeddings = get_embeddings(chunks, mode)
    results = top_chunks(query, k, chunks, bm25, embeddings, mode)
    entries[key] = results
    dump_queries(entries)
    return results
