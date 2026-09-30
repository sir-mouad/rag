import fire 
from src.indexer import indexer
from src.tokenizer import tokenizer, build_token
from src.bm25 import build_bm25
from src.load_dump import dump_chunks, dump_tokenze, load_chunks, load_tokenze, load_dataset


def index(max_chunk_size=2000):
    chunks = indexer(max_chunk_size)
    tokenized = tokenizer(chunks)
    dump_chunks(chunks)
    dump_tokenze(tokenized)
    print(f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed/")

def search(query, k=5):
    chunks = load_chunks()
    tokenized = load_tokenze()
    bm25= build_bm25(tokenized)
    query_tok = build_token(query)
    scores = bm25.get_scores(query_tok)
    top_k_indices = scores.argsort()[::-1][:k]
    for i in top_k_indices:
        chunk = chunks[i]
        print(f"{chunk['file_path']} [{chunk['first_character_index']}:{chunk['last_character_index']}]")

def search_dataset(dataset_path, k=5, save_directory="data/output/search_results"):
    question = load_dataset(dataset_path)
    


if __name__ == "__main__":
    fire.Fire()