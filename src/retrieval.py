from typing import Any, List, Optional, cast
from src.models import Chunk
from src.cache import get_index, load_queries, dump_queries
from src.embedding import load_embeddings
from src.ranking import top_chunks, MODES


def check_args(query: str, k: int) -> bool:
    """Validate the query and k, printing an error if invalid."""
    if not isinstance(k, int) or k <= 0:
        print("Invalid Argument: --k must be a positive integer")
        return False
    if not query.strip():
        print("Invalid Argument: query must not be empty")
        return False
    return True


def get_embeddings(chunks: List[Chunk], mode: str) -> Optional[Any]:
    """Return saved embeddings, or None for BM25 mode."""
    if mode not in MODES:
        raise ValueError("mode must be bm25, semantic or hybrid")
    if mode == "bm25":
        return None
    embeddings = load_embeddings(len(chunks))
    if embeddings is None:
        raise ValueError("Embeddings not found: "
                         "run index with --semantic True")
    return embeddings


def retrieve(query: str, k: int, mode: str = "bm25") -> List[Any]:
    """Return the top-k chunks for a query, using the cache if possible."""
    key = f"{mode}|{k}|{query.strip()}"
    entries = load_queries()
    if key in entries:
        return cast(List[Any], entries[key])
    chunks, bm25 = get_index()
    embeddings = get_embeddings(chunks, mode)
    results: List[Any] = top_chunks(query, k, chunks, bm25, embeddings, mode)
    entries[key] = results
    dump_queries(entries)
    return results
