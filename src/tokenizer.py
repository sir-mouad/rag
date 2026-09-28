from indexer import indexer
import re


def tokenizer(text):
    split_text = re.findall(r"[a-zA-Z0-9]+", text)
    new_split_text = []
    for str in split_text:
        new_split_text.append(str)
        split_str = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", str)
        if len(split_str) > 1:
            for sub_str in split_str:
                new_split_text.append(sub_str)
    return [str.lower() for str in new_split_text]


def main(chunks):
    chunks = indexer()
    tokenized = []
    for c in chunks:
        tokenized.append(tokenizer(c.text))

    return tokenized
