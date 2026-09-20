"""
Knowledge Base Ingestion Script for AI Travel Planning Assistant (Singapore).
Loads markdown documents from data/knowledge_base, parses metadata (source_title, source_url),
chunks them using a section-aware pure-Python markdown splitter, generates semantic embeddings
via local ONNX MiniLM, and stores them persistently in local Chroma DB.
"""

import os
import sys
import glob
import re

os.environ["ANONYMIZED_TELEMETRY"] = "False"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

KB_DIR = os.path.join(current_dir, "data", "knowledge_base")
CHROMA_DIR = os.path.join(current_dir, "chroma_db")


def parse_markdown_with_frontmatter(file_path: str):
    """Reads a markdown file, parses YAML frontmatter metadata, and returns content and dict."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    metadata = {
        "source_file": os.path.basename(file_path),
        "source_title": "Singapore Travel Knowledge",
        "source_url": "https://www.visitsingapore.com"
    }

    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if frontmatter_match:
        yaml_text = frontmatter_match.group(1)
        body = content[frontmatter_match.end():]
        for line in yaml_text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip().strip('"').strip("'")
    else:
        body = content

    return body.strip(), metadata


def split_markdown_into_chunks(text: str, max_chunk_chars: int = 700) -> list:
    """
    Splits markdown into coherent chunks respecting headers and paragraph boundaries.
    Pure Python, fast, and maintains logical context.
    """
    sections = re.split(r'\n(?=#{1,3}\s)', text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chunk_chars:
            chunks.append(sec)
        else:
            paras = sec.split('\n\n')
            current = []
            current_len = 0
            for p in paras:
                p = p.strip()
                if not p:
                    continue
                if current_len + len(p) > max_chunk_chars and current:
                    chunks.append('\n\n'.join(current))
                    current = [p]
                    current_len = len(p)
                else:
                    current.append(p)
                    current_len += len(p)
            if current:
                chunks.append('\n\n'.join(current))
    return chunks


def build_knowledge_base():
    """Ingests travel documents into persistent Chroma vector DB."""
    print("=== 1. Loading Knowledge Base Documents ===", flush=True)
    md_files = glob.glob(os.path.join(KB_DIR, "*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown documents found in {KB_DIR}")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for filepath in md_files:
        body, meta = parse_markdown_with_frontmatter(filepath)
        print(f" Loaded: {meta.get('source_title')} ({meta.get('source_file')})", flush=True)

        doc_chunks = split_markdown_into_chunks(body, max_chunk_chars=700)
        for idx, chunk in enumerate(doc_chunks):
            chunk_meta = dict(meta)
            chunk_meta["chunk_id"] = idx
            all_chunks.append(chunk)
            all_metadatas.append(chunk_meta)
            all_ids.append(f"{meta.get('source_file')}_{idx}")

    print(f"\nTotal source documents: {len(md_files)}", flush=True)
    print(f"Total chunks generated: {len(all_chunks)}", flush=True)

    print("\n=== 2. Storing in Chroma DB ===", flush=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    ef = DefaultEmbeddingFunction()
    try:
        client.delete_collection("singapore_travel_kb")
    except Exception:
        pass

    collection = client.create_collection(
        name="singapore_travel_kb",
        embedding_function=ef
    )

    batch_size = 20
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:end],
            documents=all_chunks[i:end],
            metadatas=all_metadatas[i:end]
        )
        print(f" Embedded and indexed chunks {i+1} to {end} of {len(all_chunks)}...", flush=True)

    print(f"\n Successfully stored {len(all_chunks)} chunks in Chroma DB at: {CHROMA_DIR}", flush=True)


if __name__ == "__main__":
    build_knowledge_base()
