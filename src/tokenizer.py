import re
from typing import List

from src.models import Chunk
from tqdm import tqdm


def build_token(text: str) -> List[str]:
    """Split text into lowercase tokens, including name sub-parts."""
    split_text = re.findall(r"\w+", text)
    new_split_text = []
    for str in split_text:
        new_split_text.append(str)
        parts = []
        for piece in str.split("_"):
            parts += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", piece)
        if len(parts) > 1:
            new_split_text.extend(parts)
    return [s.lower() for s in new_split_text]


def tokenizer(chunks: List[Chunk]) -> List[List[str]]:
    """Tokenize the text of every chunk and return one token list per chunk."""
    tokenized = []
    for c in tqdm(chunks, desc="tokenized", unit=" chunk"):
        tokenized.append(build_token(c.text))
    return tokenized
