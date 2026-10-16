from src.load_dump import load_history, load_chunks, load_tokenze
from src.indexer import indexer
from src.tokenizer import tokenizer
from src.models import Chunk


def check_args(query, k):
    if not isinstance(k, int) or k <= 0:
        print("Invalid Argument: --k must be a positive integer")
        return False
    if not query.strip():
        print("Invalid Argument: query must not be empty")
        return False
    return True


def get_iou(start1, end1, start2, end2):
    overlap = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    if union <= 0:
        return 0
    return overlap / union


def is_found(correct, retrieved):
    if correct["file_path"] != retrieved["file_path"]:
        return False
    iou = get_iou(correct["first_character_index"],
                  correct["last_character_index"],
                  retrieved["first_character_index"],
                  retrieved["last_character_index"])
    return iou >= 0.05


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
