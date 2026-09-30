import json
import pickle
import os


def dump_chunks(chunks):
    os.makedirs("data/processed", exist_ok=True)
    results = [chunk.model_dump() for chunk in chunks]
    with open("data/processed/chunks.json", "w") as f:
        json.dump(results, f, indent=2)


def dump_tokenze(tokenized):
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/tokenized.pkl", "wb") as f:
        pickle.dump(tokenized, f)


def load_chunks():
    with open("data/processed/chunks.json", "r") as f:
        chunks = json.load(f)
    return chunks


def load_tokenze():
    with open("data/processed/tokenized.pkl", "rb") as f:
        tokenized = pickle.load(f)
    return tokenized


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["rag_questions"]
    return [(q["question_id"], q["question"]) for q in questions]
