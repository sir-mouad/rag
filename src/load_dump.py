import json
import pickle
import os
from pathlib import Path
from pydantic import ValidationError
from typing import Any, Dict, List, Tuple, cast
from src.models import RagDataset, Chunk


def dump_chunks(chunks: List[Chunk]) -> None:
    """Save chunks to data/processed/chunks.json."""
    os.makedirs("data/processed", exist_ok=True)
    results = [chunk.model_dump() for chunk in chunks]
    with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def dump_tokenze(tokenized: List[List[str]]) -> None:
    """Save tokenized chunks to data/processed/tokenized.pkl."""
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/tokenized.pkl", "wb") as f:
        pickle.dump(tokenized, f)


def load_json(path: str) -> Any:
    """Load a JSON file, raising ValueError if missing or malformed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid json: {path} has an invalid json format")


def load_chunks() -> Any:
    """Load the saved chunks from data/processed/chunks.json."""
    if not os.path.exists("data/processed/chunks.json"):
        raise ValueError("Index not found: run the index command first")
    return load_json("data/processed/chunks.json")


def load_tokenze() -> Any:
    """Load the saved tokenized chunks from data/processed/tokenized.pkl."""
    if not os.path.exists("data/processed/tokenized.pkl"):
        raise ValueError("Index not found: run the index command first")
    with open("data/processed/tokenized.pkl", "rb") as f:
        return pickle.load(f)


def load_rag_dataset(path: str) -> RagDataset:
    """Load a JSON file and validate it as a RagDataset."""
    data = load_json(path)
    try:
        return RagDataset.model_validate(data)
    except ValidationError:
        raise ValueError(f"Invalid dataset: {path} must contain "
                         "rag_questions with question_id and question")


def load_dataset(dataset_path: str) -> List[Tuple[str, str]]:
    """Load a dataset and return (question_id, question) pairs."""
    dataset = load_rag_dataset(dataset_path)
    return [(q.question_id, q.question) for q in dataset.rag_questions]


def load_search_results(path: str) -> Dict[str, Any]:
    """Load a search results JSON file and check its fields."""
    data = load_json(path)
    if not isinstance(data, dict) or "k" not in data \
            or "search_results" not in data:
        raise ValueError(f"Invalid search results: {path} must contain "
                         "search_results and k")
    return data


def dump_json(result: Any, dataset_path: str, save_directory: str) -> Path:
    """Write a result to JSON in save_directory, keeping the input name."""
    os.makedirs(save_directory, exist_ok=True)
    filename = Path(dataset_path).name
    output_path = Path(save_directory) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    return output_path


def read_part(
    path: str,
    first_character_index: int,
    last_character_index: int,
) -> str:
    """Read a character range from a file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content[first_character_index:last_character_index]


def dump_history(files: Dict[str, List[int]], max_chunk_size: int) -> None:
    """Save file stats and chunk size to data/processed/history.json."""
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/history.json", "w", encoding="utf-8") as f:
        json.dump({"max_chunk_size": max_chunk_size, "files": files}, f)


def load_history(max_chunk_size: int) -> Dict[str, Any]:
    """Load saved file stats if the chunk size matches, else empty dict."""
    try:
        with open("data/processed/history.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["max_chunk_size"] == max_chunk_size:
            return cast(Dict[str, Any], data["files"])
    except Exception:
        pass
    return {}
