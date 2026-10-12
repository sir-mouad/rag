from rank_bm25 import BM25Okapi
from src.cache import get_index
from src.tokenizer import build_token
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.load_dump import read_part

def build_bm25(tokenized):
    return BM25Okapi(tokenized)

def build_MiniLM():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def hybrid_retrieval(query):
    chunks, bm25 = get_index()
    query_tok = build_token(query)
    bm25_scores = bm25.get_scores(query_tok)
    print(len(bm25_scores))
    print("="*40)

    model = build_MiniLM()
    texts = []
    for c in chunks:
        texts.append(c["text"])
    print(len(texts))
    chunk_embeddings = model.encode(texts)
    query_embedding = model.encode([query])
    Minilm_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    print(Minilm_scores)

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







if __name__ == "__main__":
    hybrid_retrieval("what is an llm")