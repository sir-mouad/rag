*This project has been created as part of the 42 curriculum by mhadir.*

# RAG against the machine

## Description

RAG against the machine is a Retrieval-Augmented Generation (RAG) system that answers questions about a codebase. The corpus is the vLLM 0.10.1 repository, which is much larger than what a small language model can hold in its context window and is more recent than the model's training data.

Instead of putting knowledge into the model, the system works in four stages:

1. **Indexing**: read the source files, split them into chunks, and persist a searchable index.
2. **Retrieving**: match a question against the index and return the most relevant source locations.
3. **Augmenting**: place the retrieved text into the prompt.
4. **Generating**: ask a small local model (`Qwen/Qwen3-0.6B`) to answer using only that context, and return the answer as structured JSON.

Retrieval quality is measured with recall@k: the share of correct source locations that appear in the top-k results. A result counts as correct when it is in the same file and its character range overlaps the reference span with an IoU of at least 0.05.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) as the project and package manager
- The vLLM 0.10.1 corpus placed under `data/raw/vllm-0.10.1/`

### Installation

```bash
make install
```

This runs `uv sync`, which installs the dependencies from `pyproject.toml` and `uv.lock`.

### Makefile rules

| Rule | Purpose |
| --- | --- |
| `make install` | Install dependencies with `uv sync` |
| `make run` | Run the main module (`python -m src`) |
| `make debug` | Run the main module under `pdb` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `make lint` | Run `flake8 .` and `mypy` with the required flags |
| `make lint-strict` | Run `flake8 .` and `mypy . --strict` |

### Running the pipeline

All commands are invoked as `uv run python -m src <command> [options]`.

```bash
# 1. Index the corpus (default chunk size is 2000 characters)
uv run python -m src index --max_chunk_size 2000

# 2. Search a whole dataset and save the student search results
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Score with the moulinette (rename moulinette-ubuntu/-fedora to moulinette first)
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

# 4. Generate answers from the search results
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Single-query commands

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

To use semantic or hybrid retrieval, build the index with embeddings first:

```bash
uv run python -m src index --max_chunk_size 2000 --semantic True
uv run python -m src search "How to configure the OpenAI server?" --k 5 --mode hybrid
```

### Commands

| Command | Description |
| --- | --- |
| `index --max_chunk_size <int>` | Ingest `data/raw/` and build the index under `data/processed/`. |
| `search <query> --k <int>` | Print the top-k source locations for one query. |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir> [--mode bm25\|semantic\|hybrid]` | Search every question in a dataset and write a `StudentSearchResults` JSON file. |
| `answer <query> --k <int>` | Answer one query from its retrieved context. |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Generate answers and write a `StudentSearchResultsAndAnswer` JSON file. |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Report recall@k locally for testing. The official score is computed by the moulinette. |

All input and output paths are CLI arguments and are never hard-coded. Long-running operations show progress bars with `tqdm`.

## System architecture

The pipeline is split across small modules in `src/`:

| Module | Responsibility |
| --- | --- |
| `__main__.py` | Python Fire CLI. Wires the stages together for each command. |
| `indexer.py` | Scans `data/raw/`, selects files by extension, and dispatches them to a chunker. |
| `tokenizer.py` | Turns chunk text into lowercase tokens for BM25. |
| `history.py` | Compares the current file list with the previous index to decide what to re-index. |
| `load_dump.py` | Reads and writes chunks, tokens, history, datasets, and JSON results. |
| `cache.py` | Loads the index and BM25 model, and caches query results. Invalidated when the index changes. |
| `embedding.py` | Optional semantic layer: MiniLM embeddings, persisted with the index. |
| `ranking.py` | Scores chunks with BM25, semantic similarity, or hybrid RRF fusion. |
| `retrieval.py` | Validates arguments and runs a single query through the ranking step, with caching. |
| `generator.py` | Builds the prompt and generates the answer with `Qwen/Qwen3-0.6B`. |
| `evaluate.py` | Computes IoU and whether a retrieved source counts as a match. |
| `models.py` | Pydantic models for every structure exchanged between stages. |

Data flows in one direction:

```
data/raw/ -> indexer -> tokenizer -> index (chunks.json, tokenized.pkl, BM25)
                                         |
question -> retrieval -> ranking (BM25 / semantic / hybrid) -> top-k sources
                                         |
                    read_part / chunk text -> generator (Qwen3-0.6B) -> answer JSON
```

Every output is validated through pydantic models (`MinimalSource`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`) before it is written.

## Chunking strategy

The corpus has two kinds of files, so the indexer uses two chunking strategies.

**Python files (`.py`)** are parsed with the standard `ast` module. Chunk boundaries are placed at the start of top-level statements, so functions and classes are not split in the middle when possible. Consecutive statements are merged into one chunk while they fit within `max_chunk_size`. A section that is still too long is split line by line. If a file cannot be parsed, it falls back to the Markdown strategy.

**Markdown and text files (`.md`, `.txt`)** are split at header lines (`#`, `##`, ...). Sections are merged while they fit. Long sections are split at line boundaries.

In both cases:

- The default chunk size is 2000 characters, set with `--max_chunk_size`. No chunk exceeds it, because the evaluator rejects sources longer than `max_context_length`.
- Each chunk records its `file_path` (relative to the project root, exactly as in the corpus) and its `first_character_index` and `last_character_index`.
- Chunks that are empty or only whitespace are discarded.

Smaller chunks are accepted. The effect of chunk size on recall is reported in the Performance analysis section.

## Retrieval method

### Lexical retrieval (BM25, default)

Each chunk is tokenized by `tokenizer.py`. The tokenizer keeps every word as a token and also adds its parts when an identifier contains `_` or CamelCase boundaries. For example, `load_chunks` yields `load_chunks`, `load`, and `chunks`. This helps when a question uses a short name for an identifier that appears in the code under a longer one. All tokens are lowercased.

Scoring uses `rank_bm25.BM25Okapi`. A query is tokenized the same way and scored against every chunk. The top-k chunks by score are returned. If no query token matches any chunk, the result is empty.

### Semantic retrieval

When the index is built with semantic embeddings, every chunk is encoded with `sentence-transformers/all-MiniLM-L6-v2` and normalized. At query time the query is encoded the same way, and the score is the dot product between the query vector and each chunk vector (cosine similarity).

### Hybrid retrieval

In `hybrid` mode, the BM25 and semantic scores are combined with reciprocal rank fusion (RRF). Each chunk gets `1 / (60 + rank_bm25) + 1 / (60 + rank_semantic)`, and the chunks are sorted by that sum. If BM25 finds no match for a query, hybrid mode falls back to the semantic ranking.

### Caching

- The index (chunks and BM25 model) is pickled to `data/cache/index.pkl`. It is reloaded only if the modification time and size of the chunk and token files have not changed.
- Query results are stored in `data/cache/queries.json`, keyed by mode, k, and query. The cache is tied to the same index signature.
- Running `index` again removes the cache directory.

### Incremental indexing

`index` stores the modification time and size of every corpus file in `data/processed/history.json`. On the next run, only files whose stats changed are re-chunked and re-tokenized. Chunks from unchanged files are reused from the previous index. The chunk size is part of the history, so changing `--max_chunk_size` forces a full rebuild.

## Bonus features

- **Semantic embeddings** (`index --semantic True`): builds a vector index next to the lexical index. Required before using `semantic` or `hybrid` mode.
- **Hybrid retrieval** (`--mode hybrid` on `search`, `answer`, or `search_dataset`): combines BM25 and semantic rankings with RRF.
- **Incremental indexing**: re-indexes only the files that changed.
- **Caching**: index and query caches for faster cold starts and repeated queries.

The local HTTP API bonus is not implemented.

## Design decisions

- **Two chunking strategies** because a Python file and a Markdown page have different natural boundaries. Splitting code at definitions keeps related logic together.
- **BM25 as the default** because it is fast on CPU, needs no model download, and handles exact identifiers well. Questions often quote function or class names verbatim.
- **Identifier splitting in the tokenizer** to bridge the gap between natural-language questions and code identifiers.
- **Pydantic for exchanged data** so every stage validates its input and output. Services such as the indexer and retriever are plain classes or functions.
- **Character ranges instead of line numbers** because the evaluator and the answer generator both work with character offsets.
- **Caching keyed by an index signature** so a rebuilt index never serves stale results.

## Performance analysis

Recall figures depend on the reference dataset and the chunk size. Measure them on your own run with:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

The official thresholds are 80 % recall@5 on docs questions and 50 % on code questions. Indexing the full corpus must finish within 5 minutes, and 200 retrieval queries must finish within 90 seconds.

The recall values, timings, and the effect of `--max_chunk_size` should be recorded here after the final run.

## Challenges faced

- **Questions and code use different words.** Paraphrased questions rarely match identifiers exactly. Identifier splitting in the tokenizer addresses part of this.
- **Chunk size trades off context and precision.** Large chunks contain the answer but add noise; small chunks are precise but can split an answer across two chunks.
- **Keeping the index fresh without rebuilding everything.** The history file records file stats so unchanged files are not re-processed.
- **Output validity.** Every source must have a path that matches the corpus exactly and a range under the context limit, so chunking is bounded by `max_chunk_size`.

## Example usage

```bash
# Build the index
uv run python -m src index --max_chunk_size 2000

# Find the sources for a question
uv run python -m src search "How to configure the OpenAI server?" --k 5

# Answer a question
uv run python -m src answer "How to configure the OpenAI server?" --k 5

# Run the full evaluation pipeline
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```
