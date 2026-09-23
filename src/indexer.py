from pathlib import Path
from models import Chunk
# import re


def chunk_python(path, content, max_chunk_size):
    if len(content) <= max_chunk_size:
        return [Chunk(
            file_path=path,
            first_character_index=0,
            last_character_index=len(content),
            text=content
        )]
    


def chunk_markdown(path, content, max_chunk_size):
    if len(content) <= max_chunk_size:
        return [Chunk(
            file_path=path,
            first_character_index=0,
            last_character_index=len(content),
            text=content
        )]
    for line in content:
        if line.startwith("#"):


def load_and_chunk_file(path, max_chunk_size):
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(e)
        exit(1)
    if path.suffix == ".py":
        chunk_python(str(path), content, max_chunk_size)
    elif path.suffix == ".md":
        chunk_markdown(str(path),
                       content, max_chunk_size)


def main() -> None:

    cwd = Path.cwd()
    raw = cwd / "data" / "raw" / "vllm-0.10.1"
    paths = [f for f in raw.rglob("*") if f.suffix in (".py", ".md")]
    max_chunk_size = 2000
    chunks = []
    for path in paths:
        chunks.append(load_and_chunk_file(path, max_chunk_size))


if __name__ == "__main__":
    main()
