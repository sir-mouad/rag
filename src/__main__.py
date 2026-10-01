import fire
from src.indexer import indexer
from src.tokenizer import tokenizer, build_token
from src.bm25 import build_bm25
from src.load_dump import dump_chunks, dump_tokenze, load_chunks, load_tokenze, load_dataset, dump_search_results
from tqdm import tqdm
from src.models import MinimalSource, MinimalSearchResults, StudentSearchResults


def index(max_chunk_size=2000):
    chunks = indexer(max_chunk_size)
    tokenized = tokenizer(chunks)
    dump_chunks(chunks)
    dump_tokenze(tokenized)
    print(
        f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed/")


def search(query, k=5):
    chunks = load_chunks()
    tokenized = load_tokenze()
    bm25 = build_bm25(tokenized)
    query_tok = build_token(query)
    scores = bm25.get_scores(query_tok)
    top_k_indices = scores.argsort()[::-1][:k]
    for i in top_k_indices:
        chunk = chunks[i]
        print(
            f"{chunk['file_path']} [{chunk['first_character_index']}:{chunk['last_character_index']}]")


def search_dataset(dataset_path, k=5, save_directory="data/output/search_results"):
    chunks = load_chunks()
    tokenized = load_tokenze()
    bm25 = build_bm25(tokenized)
    questions = load_dataset(dataset_path)
    search_results = []
    for ques_id, ques in tqdm(questions):
        ques_tok = build_token(ques)
        scores = bm25.get_scores(ques_tok)
        top_k_indices = scores.argsort()[::-1][:k]
        sources = []
        for i in top_k_indices:
            chunk = chunks[i]
            sources.append(MinimalSource(file_path=chunk["file_path"],
                                         first_character_index=chunk["first_character_index"],
                                         last_character_index=chunk["last_character_index"]))
        search_results.append(MinimalSearchResults(
            question_id=ques_id,
            question=ques,
            retrieved_sources=sources
        ))

    result = StudentSearchResults(search_results=search_results, k=k)
    dump_search_results(result, dataset_path, save_directory)


if __name__ == "__main__":
    fire.Fire()
