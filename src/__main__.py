import fire
from src.indexer import indexer
from src.tokenizer import tokenizer, build_token
from src.bm25 import build_bm25
from src.load_dump import dump_chunks, dump_tokenze, load_chunks, load_tokenze, load_dataset, dump_search_results, load_search_results, read_part
from tqdm import tqdm
from src.models import MinimalSource, MinimalSearchResults, StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer
from src.generator import load_model, build_prompt, get_answer_text
import uuid
import re


def index(max_chunk_size=2000):
    chunks = indexer(max_chunk_size)
    tokenized = tokenizer(chunks)
    dump_chunks(chunks)
    dump_tokenze(tokenized)
    print(
        f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed/")


def search(query, k):
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


def search_dataset(dataset_path, k, save_directory="data/output/search_results"):
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
    output_path = dump_search_results(result, dataset_path, save_directory)
    print(f"Saved student_search_results to {output_path}")


def answer(query, k):
    llm_tokenizer, model = load_model()
    chunks = load_chunks()
    tokenized = load_tokenze()
    bm25 = build_bm25(tokenized)
    query_tok = build_token(query)
    scores = bm25.get_scores(query_tok)
    top_k_indices = scores.argsort()[::-1][:k]
    sources = []
    texts = []
    for i in top_k_indices:
        chunk = chunks[i]
        sources.append(MinimalSource(file_path=chunk["file_path"],
                                     first_character_index=chunk["first_character_index"],
                                     last_character_index=chunk["last_character_index"]))
        texts.append(chunk["text"])
    content = "\n\n".join(texts)
    prompt = build_prompt(query, content)
    answer_text = get_answer_text(prompt, llm_tokenizer, model)
    result = MinimalAnswer(
        question_id=str(uuid.uuid4()),
        question=query,
        retrieved_sources=sources,
        answer=answer_text
    )

    print(result.model_dump_json(indent=2))


def answer_dataset(student_search_results_path, save_directory):
    llm_tokenizer, model = load_model()
    search_results = load_search_results(student_search_results_path)
    k = search_results["k"]
    results = []
    for res in tqdm(search_results["search_results"], desc="Answering", unit="question"):
        ques_id = res["question_id"]
        ques = res["question"]
        sources = res["retrieved_sources"]
        texts = []
        for s in sources:
            texts.append(read_part(
                s["file_path"], s["first_character_index"], s["last_character_index"]))
        content = "\n\n".join(texts)
        prompt = build_prompt(ques, content)
        answer_text = get_answer_text(prompt, llm_tokenizer, model)
        results.append(MinimalAnswer(
            question_id=ques_id,
            question=ques,
            retrieved_sources=sources,
            answer=answer_text
        ))

    final_results = StudentSearchResultsAndAnswer(
        search_results=results,
        k=k
    )
    output_path = dump_search_results(
        final_results, student_search_results_path, save_directory)
    print(f"Saved student_search_results_and_answer to {output_path}")


if __name__ == "__main__":
    fire.Fire()
