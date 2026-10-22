from typing import Any, Dict, List, Tuple, cast

import json
import os
import pickle
from rank_bm25 import BM25Okapi
from src.load_dump import load_chunks, load_tokenze
from src.models import Chunk

CACHE_DIR = "data/cache"
INDEX_CACHE = "data/cache/index.pkl"
QUERY_CACHE = "data/cache/queries.json"
CHUNKS_PATH = "data/processed/chunks.json"
TOKENIZED_PATH = "data/processed/tokenized.pkl"


def build_bm25(tokenized: List[List[str]]) -> BM25Okapi:
    """Build a BM25 index from tokenized chunks."""
    return BM25Okapi(tokenized)


def index_signature() -> List[List[int]]:
    """Return mtime and size of the chunk and token files for change checks."""
    signature = []
    for path in (CHUNKS_PATH, TOKENIZED_PATH):
        info = os.stat(path)
        signature.append([info.st_mtime_ns, info.st_size])
    return signature


def get_index() -> Tuple[List[Chunk], BM25Okapi]:
    """Return chunks and BM25 index, using the cache when still valid."""
    try:
        signature = index_signature()
        with open(INDEX_CACHE, "rb") as f:
            cache = pickle.load(f)
        if cache["signature"] == signature:
            return cache["chunks"], cache["bm25"]
    except Exception:
        pass
    chunks = load_chunks()
    tokenized = load_tokenze()
    bm25 = build_bm25(tokenized)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = INDEX_CACHE + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"signature": index_signature(), "chunks": chunks,
                         "bm25": bm25}, f)
        os.replace(tmp, INDEX_CACHE)
    except Exception:
        pass
    return chunks, bm25


def load_queries() -> Dict[str, Any]:
    """Return cached query results if they still match the current index."""
    try:
        with open(QUERY_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["signature"] == index_signature():
            return cast(Dict[str, Any], data["entries"])
    except Exception:
        pass
    return {}


def dump_queries(entries: Dict[str, Any]) -> None:
    """Save query results to the cache with the current index signature."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(QUERY_CACHE, "w", encoding="utf-8") as f:
            json.dump({"signature": index_signature(), "entries": entries}, f)
    except Exception:
        pass
