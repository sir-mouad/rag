import os
import pickle
from src.load_dump import load_chunks, load_tokenze
from src.bm25 import build_bm25

CACHE_DIR = "data/cache"
INDEX_CACHE = "data/cache/index.pkl"
CHUNKS_PATH = "data/processed/chunks.json"
TOKENIZED_PATH = "data/processed/tokenized.pkl"


def index_stamp():
    stamp = []
    for path in (CHUNKS_PATH, TOKENIZED_PATH):
        info = os.stat(path)
        stamp.append((info.st_mtime_ns, info.st_size))
    return stamp


def get_index():
    try:
        stamp = index_stamp()
        with open(INDEX_CACHE, "rb") as f:
            cache = pickle.load(f)
        if cache["stamp"] == stamp:
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
            pickle.dump({"stamp": index_stamp(), "chunks": chunks,
                         "bm25": bm25}, f)
        os.replace(tmp, INDEX_CACHE)
    except Exception:
        pass
    return chunks, bm25