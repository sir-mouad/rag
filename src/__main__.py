import fire
from src.indexer import indexer
from src.tokenizer import tokenizer, build_token
from src.bm25 import build_bm25
from src.load_dump import dump_chunks, dump_tokenze, load_chunks, load_tokenze, load_dataset, dump_search_results, load_search_results, read_part
from tqdm import tqdm
from src.models import MinimalSource, MinimalSearchResults, StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer
from src.generator import load_model, build_prompt, get_answer_text
from src.evaluate import evaluate
import uuid


def check_args(query, k):
    if not isinstance(k, int) or k <= 0:
        print("Invalid Argument: --k must be a positive integer")
        return False
    if not query.strip():
        print("Invalid Argument: query must not be empty")
        return False
    return True


def index(max_chunk_size=2000):
    try:
        if not isinstance(max_chunk_size, int) or not 1 <= max_chunk_size <= 2000:
            print("Invalid Argument: --max_chunk_size must be an integer between 1 and 2000")
            return
        chunks = indexer(max_chunk_size)
        if not chunks:
            print("Error: no chunks were created, nothing to index.")
            return
        tokenized = tokenizer(chunks)
        dump_chunks(chunks)
        dump_tokenze(tokenized)
        print(
            f"Ingestion complete! Indexed {len(chunks)} chunks under data/processed/")
    except Exception as e:
        print(e)


def search(query, k):
    try:
        query = str(query)
        if not check_args(query, k):
            return
        chunks = load_chunks()
        tokenized = load_tokenze()
        bm25 = build_bm25(tokenized)
        query_tok = build_token(query)
        scores = bm25.get_scores(query_tok)
        if not query_tok or scores.max() <= 0:
            print("No relevant results found for this query.")
            return
        top_k_indices = scores.argsort()[::-1][:k]
        for i in top_k_indices:
            chunk = chunks[i]
            print(
                f"{chunk['file_path']} [{chunk['first_character_index']}:{chunk['last_character_index']}]")
    except Exception as e:
        print(e)


def search_dataset(dataset_path, k, save_directory):
    try:
        if not isinstance(k, int) or k <= 0:
            print("Invalid Argument: --k must be a positive integer")
            return
        chunks = load_chunks()
        tokenized = load_tokenze()
        bm25 = build_bm25(tokenized)
        questions = load_dataset(dataset_path)
        search_results = []
        for ques_id, ques in tqdm(questions, desc="searching", unit="question"):
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
    except Exception as e:
        print(e)


def answer(query, k):
    try:
        query = str(query)
        if not check_args(query, k):
            return
        chunks = load_chunks()
        tokenized = load_tokenze()
        bm25 = build_bm25(tokenized)
        query_tok = build_token(query)
        scores = bm25.get_scores(query_tok)
        if not query_tok or scores.max() <= 0:
            print("No relevant results found for this query.")
            return
        llm_tokenizer, model = load_model()
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
    except Exception as e:
        print(e)


def answer_dataset(student_search_results_path, save_directory):
    try:
        search_results = load_search_results(student_search_results_path)
        k = search_results["k"]
        llm_tokenizer, model = load_model()
        results = []
        for res in tqdm(search_results["search_results"], desc="Answering", unit="question"):
            ques_id = res["question_id"]
            ques = res["question"]
            sources = res["retrieved_sources"]
            try:
                texts = []
                for s in sources:
                    texts.append(read_part(
                        s["file_path"], s["first_character_index"], s["last_character_index"]))
                if not texts:
                    answer_text = "I don't know."
                else:
                    content = "\n\n".join(texts)
                    prompt = build_prompt(ques, content)
                    answer_text = get_answer_text(
                        prompt, llm_tokenizer, model)
            except Exception as e:
                print(f"Warning: question {ques_id} failed: {e}")
                answer_text = "I don't know."
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
    except Exception as e:
        print(e)


if __name__ == "__main__":
    fire.Fire()