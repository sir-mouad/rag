import fire
import os
import shutil
from tqdm import tqdm
from src.indexer import scan_files
from src.retrieval import retrieve, get_embeddings, check_args
from src.ranking import top_chunks
from src.cache import get_index, load_queries, dump_queries
from src.embedding import dump_embeddings, EMBEDDINGS_PATH, update_embeddings
from src.load_dump import dump_chunks, dump_tokenze, dump_history, dump_json, \
    read_part, load_dataset, load_json, \
    load_search_results, load_rag_dataset
from src.history import merge_index, load_old_index
from src.evaluate import is_found
from src.generator import load_model, build_prompt, get_answer_text
from src.models import MinimalSource, MinimalSearchResults, \
    StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer, \
    AnsweredQuestion, UnansweredQuestion


def index(max_chunk_size=2000, semantic=False):
    try:
        if not isinstance(max_chunk_size, int) or \
                max_chunk_size <= 0:
            print(
                "Invalid Argument: --max_chunk_size must be an "
                "integer positif")
            return
        files = scan_files()
        if not files:
            print("Error: no files found, nothing to index.")
            return
        old = load_old_index(max_chunk_size)
        chunks, tokenized, new_chunks, kept, changed = merge_index(
            files, old, max_chunk_size)
        if not chunks:
            print("Error: no chunks were created, nothing to index.")
            return
        embeddings = None
        if semantic:
            embeddings = update_embeddings(old[1], kept, new_chunks, chunks)
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
        dump_chunks(chunks)
        dump_tokenze(tokenized)
        dump_history(files, max_chunk_size)
        shutil.rmtree("data/cache", ignore_errors=True)
        if embeddings is not None:
            dump_embeddings(embeddings)
        print(
            f"Ingestion complete! Indexed {len(chunks)} chunks "
            "under data/processed/ ")
        if kept:
            print(f"({len(changed)} files re-indexed)")
    except Exception as e:
        print(e)


def search(query, k, mode="bm25"):
    try:
        query = str(query)
        if not check_args(query, k):
            return
        results = retrieve(query, k, mode)
        if not results:
            print("No relevant results found for this query.")
            return
        for chunk in results:
            print(
                f"{chunk['file_path']} [{chunk['first_character_index']}:"
                f"{chunk['last_character_index']}]")
    except Exception as e:
        print(e)


def search_dataset(dataset_path, k, save_directory, mode="bm25"):
    try:
        if not isinstance(k, int) or k <= 0:
            print("Invalid Argument: --k must be a positive integer")
            return
        chunks, bm25 = get_index()
        embeddings = get_embeddings(chunks, mode)
        questions = load_dataset(dataset_path)
        search_results = []
        for ques_id, ques in tqdm(questions, desc="searching",
                                  unit="question"):
            sources = []
            for chunk in top_chunks(ques, k, chunks, bm25, embeddings, mode):
                sources.append(MinimalSource(file_path=chunk["file_path"],
                                             first_character_index=chunk[
                    "first_character_index"],
                    last_character_index=chunk[
                    "last_character_index"]))
            search_results.append(MinimalSearchResults(
                question_id=ques_id,
                question=ques,
                retrieved_sources=sources
            ))

        result = StudentSearchResults(search_results=search_results, k=k)
        output_path = dump_json(result, dataset_path, save_directory)
        print(f"Saved student_search_results to {output_path}")
    except Exception as e:
        print(e)


def answer(query, k, mode="bm25"):
    try:
        query = str(query)
        if not check_args(query, k):
            return
        question = UnansweredQuestion(question=query)
        key = f"answer|{mode}|{k}|{query.strip()}"
        entries = load_queries()
        if key in entries:
            print(MinimalAnswer(**entries[key]).model_dump_json(indent=2))
            return
        results = retrieve(query, k, mode)
        if not results:
            print("No relevant results found for this query.")
            return
        llm_tokenizer, model = load_model()
        sources = []
        texts = []
        for chunk in results:
            sources.append(MinimalSource(file_path=chunk["file_path"],
                                         first_character_index=chunk[
                "first_character_index"],
                last_character_index=chunk[
                "last_character_index"]))
            texts.append(chunk["text"])
        content = "\n\n".join(texts)
        prompt = build_prompt(query, content)
        answer_text = get_answer_text(prompt, llm_tokenizer, model)
        result = MinimalAnswer(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
            answer=answer_text
        )
        entries = load_queries()
        entries[key] = result.model_dump()
        dump_queries(entries)
        print(result.model_dump_json(indent=2))
    except Exception as e:
        print(e)


def answer_dataset(student_search_results_path, save_directory):
    try:
        search_results = load_search_results(student_search_results_path)
        k = search_results["k"]
        llm_tokenizer, model = load_model()
        results = []
        for res in tqdm(search_results["search_results"], desc="Answering",
                        unit="question"):
            ques_id = res["question_id"]
            ques = res["question"]
            sources = res["retrieved_sources"]
            try:
                texts = []
                for s in sources:
                    texts.append(read_part(
                        s["file_path"], s["first_character_index"],
                        s["last_character_index"]))
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
            dump_json(StudentSearchResultsAndAnswer(
                search_results=results,
                k=k
            ), student_search_results_path, save_directory)

        final_results = StudentSearchResultsAndAnswer(
            search_results=results,
            k=k
        )
        output_path = dump_json(
            final_results, student_search_results_path, save_directory)
        print(f"Saved student_search_results_and_answer to {output_path}")
    except Exception as e:
        print(e)


def evaluate(student_search_results_path, dataset_path):
    try:
        search_results = load_json(student_search_results_path)
        truth = load_rag_dataset(dataset_path)
        correct_sources = {q.question_id: [s.model_dump() for s in q.sources]
                           for q in truth.rag_questions
                           if isinstance(q, AnsweredQuestion)}
        ks = [1, 3, 5, 10]
        totals = {k: 0 for k in ks}
        counted = 0
        for res in tqdm(search_results["search_results"],
                        desc="Evaluating", unit="question"):
            correct = correct_sources.get(res["question_id"])
            if not correct:
                continue
            counted += 1
            for k in ks:
                top = res["retrieved_sources"][:k]
                found = 0
                for c in correct:
                    if any(is_found(c, r) for r in top):
                        found += 1
                totals[k] += found / len(correct)
        if counted == 0:
            print("Error: no question could be matched with the ground truth.")
            return
        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {counted}")
        final = {}
        for k in ks:
            recall = totals[k] / counted
            print(f"Recall@{k}: {recall:.3f} ({recall * 100:.1f}%)")
            final[f"Recall@{k}"] = f"{recall:g}"
        print(final)
    except (KeyError, TypeError):
        print("Invalid file format: missing fields in the files")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate
    })
