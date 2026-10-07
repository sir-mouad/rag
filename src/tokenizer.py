import re
from tqdm import tqdm


def build_token(text):
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


def tokenizer(chunks):
    tokenized = []
    for c in tqdm(chunks, desc="tokenized", unit=" chunk"):
        tokenized.append(build_token(c.text))
    return tokenized
