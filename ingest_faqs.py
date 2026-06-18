#!/usr/bin/env python3
"""
Ingest FAQ documents (markdown/text) into a FAISS vector store using OpenAI embeddings.

Usage:
  python ingest_faqs.py --source ./faqs --output ./vector_store

Requirements:
  - Set OPENAI_API_KEY in environment or .env
  - Installed requirements (faiss-cpu, numpy, openai)
"""
import os
import json
import argparse
import uuid
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file")
openai.api_key = OPENAI_API_KEY

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE_CHARS", "1000"))  # characters per chunk
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "200"))


def read_files(source_dir: Path) -> List[Path]:
    files = []
    for ext in ("*.md", "*.txt"):
        files.extend(list(source_dir.rglob(ext)))
    return files


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def embed_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    # OpenAI embeddings API supports batching multiple inputs in one call.
    # For large datasets you may want to batch in smaller sizes.
    resp = openai.Embedding.create(model=model, input=texts)
    embeddings = [r["embedding"] for r in resp["data"]]
    return embeddings


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    # Normalize embeddings for cosine similarity using inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory with .md/.txt files to index")
    parser.add_argument("--output", required=True, help="Output directory for vector_store")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    files = read_files(src)
    if not files:
        print("No files found in", src)
        return

    docs = []  # list of metadata dicts
    texts_to_embed = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            docs.append({
                "id": doc_id,
                "source": str(f.relative_to(src)),
                "chunk_index": i,
                "text": chunk,
            })
            texts_to_embed.append(chunk)

    # Create embeddings in batches
    all_embeddings = []
    total = len(texts_to_embed)
    bs = args.batch_size
    for i in range(0, total, bs):
        batch = texts_to_embed[i : i + bs]
        embs = embed_batch(batch)
        all_embeddings.extend(embs)
        print(f"Embedded {i + len(batch)}/{total}")

    emb_array = np.array(all_embeddings).astype("float32")
    # Normalize before storing
    faiss.normalize_L2(emb_array)

    index = build_index(emb_array)

    # Save index and docs metadata
    index_path = out / "index.faiss"
    faiss.write_index(index, str(index_path))

    docs_path = out / "docs.json"
    with docs_path.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)

    print("Saved FAISS index to", index_path)
    print("Saved docs metadata to", docs_path)
    print("Done.")


if __name__ == "__main__":
    main()
