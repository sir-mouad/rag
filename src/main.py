from indexer import indexer
from tokenizer import main
from bm25 import build_bm25


chunks = indexer()
tokenized = main(chunks)
build_bm25(tokenized)
