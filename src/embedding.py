import os
import numpy as np

EMBEDDINGS_PATH = "data/processed/embeddings.npy"
minilm = None


def build_MiniLM():
    global minilm
    if minilm is None:
        from sentence_transformers import SentenceTransformer
        minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return minilm


def build_embeddings(texts):
    return build_MiniLM().encode(texts, batch_size=64,
                                 show_progress_bar=True,
                                 normalize_embeddings=True)


def dump_embeddings(embeddings):
    os.makedirs("data/processed", exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)


def load_embeddings(n_chunks):
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
    except Exception:
        return None
    if embeddings.shape[0] != n_chunks:
        return None
    return embeddings


def semantic_scores(query, embeddings):
    query_vector = build_MiniLM().encode([query],
                                         normalize_embeddings=True)[0]
    return embeddings @ query_vector


def update_embeddings(old_chunks, kept, new_chunks, chunks):
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
