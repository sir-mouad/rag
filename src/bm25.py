from rank_bm25 import BM25Okapi


def build_bm25(tokenized):
    bm25 = BM25Okapi(tokenized)
    print(bm25)
