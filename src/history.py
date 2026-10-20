from src.load_dump import load_history, load_chunks, load_tokenze
from src.indexer import indexer
from src.tokenizer import tokenizer
from src.models import Chunk


def load_old_index(max_chunk_size):
    old_files = load_history(max_chunk_size)
    if not old_files:
        return {}, [], []
    old_chunks = []
    try:
        for c in load_chunks():
            old_chunks.append(Chunk(**c))
        old_tokenized = load_tokenze()
        if len(old_chunks) != len(old_tokenized):
            raise ValueError("chunks.json and tokenized.pkl have a "
                             "different number of entries")
    except (OSError, ValueError, TypeError, KeyError, EOFError) as e:
        print(f"Previous index unusable ({e}), rebuilding everything")
        return {}, [], []
    return old_files, old_chunks, old_tokenized


def merge_index(files, old, max_chunk_size):
    old_files, old_chunks, old_tokenized = old
    changed = set()
    for path, stat in files.items():
        if old_files.get(path) != stat:
            changed.add(path)
    kept = []
    for i, c in enumerate(old_chunks):
        if c.file_path in files and c.file_path not in changed:
            kept.append(i)
    new_chunks = indexer(max_chunk_size, changed)
    new_tokenized = tokenizer(new_chunks)
    chunks = []
    tokenized = []
    for i in kept:
        chunks.append(old_chunks[i])
        tokenized.append(old_tokenized[i])
    chunks.extend(new_chunks)
    tokenized.extend(new_tokenized)
    return chunks, tokenized, new_chunks, kept, changed
