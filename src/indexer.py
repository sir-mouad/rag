from pathlib import Path
from src.models import Chunk
import re
import ast
from tqdm import tqdm


def chunk_python(path, content, max_chunk_size):
    if len(content) <= max_chunk_size:
        return [Chunk(
            file_path=path,
            first_character_index=0,
            last_character_index=len(content),
            text=content
        )]

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_markdown(path, content, max_chunk_size)

    lines = content.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    headers = [0]
    for node in tree.body:
        start = line_offsets[node.lineno - 1]
        if start != 0:
            headers.append(start)
    headers.append(len(content))

    true_headers = [0]
    index = 0
    while index + 1 < len(headers):
        h_range = headers[index + 1] - true_headers[-1]
        if h_range <= max_chunk_size:
            index += 1
        else:
            true_headers.append(headers[index])
            index += 1
    if true_headers[-1] != len(content):
        true_headers.append(len(content))

    chunks = []
    for start, end in zip(true_headers, true_headers[1:]):
        section = content[start:end]
        if not section.strip():
            continue
        if len(section) <= max_chunk_size:
            chunks.append(Chunk(file_path=path,
                                first_character_index=start,
                                last_character_index=end,
                                text=section))
            continue
        pos = start
        current_start = start
        for line in section.splitlines(keepends=True):
            if pos + len(line) - current_start > max_chunk_size and pos != current_start:
                section = content[current_start:pos]
                if section.strip():
                    chunks.append(Chunk(file_path=path,
                                        first_character_index=current_start,
                                        last_character_index=pos,
                                        text=section))
                current_start = pos
            pos += len(line)
        section = content[current_start:pos]
        if section.strip():
            chunks.append(Chunk(file_path=path,
                                first_character_index=current_start,
                                last_character_index=pos,
                                text=section))
    return chunks


def chunk_markdown(path, content, max_chunk_size):
    if len(content) <= max_chunk_size:
        return [Chunk(
            file_path=path,
            first_character_index=0,
            last_character_index=len(content),
            text=content
        )]

    match = re.findall(r"(?m)^#+ [^\r\n]*", content)

    headers = [0]
    offset = 0
    for line in content.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if stripped in match and offset != 0:
            headers.append(offset)
        offset += len(line)
    headers.append(len(content))

    true_headers = [0]
    index = 0
    while index + 1 < len(headers):
        h_range = headers[index + 1] - true_headers[-1]
        if h_range <= max_chunk_size:
            index += 1
        else:
            true_headers.append(headers[index])
            index += 1
    if true_headers[-1] != len(content):
        true_headers.append(len(content))
    chunks = []
    for start, end in zip(true_headers, true_headers[1:]):
        section = content[start:end]
        if len(section) <= max_chunk_size:
            if not section.strip():
                continue
            chunks.append(Chunk(file_path=path, first_character_index=start,
                          last_character_index=end, text=section))
            continue
        pos = start
        current_start = start
        for line in section.splitlines(keepends=True):
            if pos + len(line) - current_start > max_chunk_size and pos != current_start:
                chunks.append(Chunk(file_path=path, first_character_index=current_start, last_character_index=pos,
                              text=content[current_start:pos]))
                current_start = pos
            pos += len(line)
        chunks.append(Chunk(file_path=path,
                            first_character_index=current_start,
                            last_character_index=pos,
                      text=content[current_start:pos]))

    return chunks


def load_and_chunk_file(path, max_chunk_size, cwd):
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return []
    if not content.strip():
        return []
    if path.suffix == ".py":
        return (chunk_python(str(path.relative_to(cwd)), content, max_chunk_size))
    elif path.suffix == ".md" or  path.suffix == ".txt":
        return (chunk_markdown(str(path.relative_to(cwd)),
                               content, max_chunk_size))


def indexer(max_chunk_size):

    cwd = Path.cwd()
    raw = cwd / "data" / "raw" / "vllm-0.10.1"
    if not raw.is_dir():
        print(f"Error: directory not found: {raw}")
        return []
    paths = [f for f in raw.rglob("*") if f.suffix in (".py", ".md", ".txt")]
    chunks = []
    for path in tqdm(list(paths), desc="Chunking", unit=" file"):
        chunks.extend(load_and_chunk_file(
            path, max_chunk_size, cwd))

    return chunks
