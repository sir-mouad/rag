import json
import pickle
import os
from pathlib import Path


def dump_chunks(chunks):
    os.makedirs("data/processed", exist_ok=True)
    results = [chunk.model_dump() for chunk in chunks]
    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def dump_tokenze(tokenized):
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/tokenized.pkl", "wb") as f:
        pickle.dump(tokenized, f)


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid json: {path} has an invalid json format")


def load_chunks():
    if not os.path.exists("data/processed/chunks.json"):
        raise ValueError("Index not found: run the index command first")
    return load_json("data/processed/chunks.json")


def load_tokenze():
    if not os.path.exists("data/processed/tokenized.pkl"):
        raise ValueError("Index not found: run the index command first")
    with open("data/processed/tokenized.pkl", "rb") as f:
        return pickle.load(f)


def load_dataset(dataset_path):
    data = load_json(dataset_path)
    try:
        questions = data["rag_questions"]
        return [(q["question_id"], q["question"]) for q in questions]
    except (KeyError, TypeError):
        raise ValueError(f"Invalid dataset: {dataset_path} must contain "
                         "rag_questions with question_id and question")


def load_search_results(path):
    data = load_json(path)
    if not isinstance(data, dict) or "k" not in data \
            or "search_results" not in data:
        raise ValueError(f"Invalid search results: {path} must contain "
                         "search_results and k")
    return data


def dump_json(result, dataset_path, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    filename = Path(dataset_path).name
    output_path = Path(save_directory) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    return output_path


def read_part(path, first_character_index, last_character_index):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content[first_character_index:last_character_index]


def dump_history(files, max_chunk_size):
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/history.json", "w", encoding="utf-8") as f:
        json.dump({"max_chunk_size": max_chunk_size, "files": files}, f)


def load_history(max_chunk_size):
    try:
        with open("data/processed/history.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["max_chunk_size"] == max_chunk_size:
            return data["files"]
    except Exception:
        pass
    return {}
