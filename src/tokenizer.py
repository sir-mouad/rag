import re
from tqdm import tqdm

def build_token(text):
    split_text = re.findall(r"[a-zA-Z0-9]+", text)
    new_split_text = []
    for str in split_text:
        new_split_text.append(str)
        split_str = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str)
        if len(split_str) > 1:
            for sub_str in split_str:
                new_split_text.append(sub_str)
    return [str.lower() for str in new_split_text]


def tokenizer(chunks):
    tokenized = []
    for c in tqdm(chunks, desc="tokenized", unit=" chunk"):
        tokenized.append(build_token(c.text))

    return tokenized
