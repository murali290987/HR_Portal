"""
ingest.py — Steps 2 & 3.

Step 2: load HR docs, split into overlapping chunks, tag each with
metadata (doc_type, employee_id, etc).

Step 3: turn each chunk's text into a vector embedding, store all
vectors in a FAISS index for fast similarity search, and save both the
index and a metadata store to disk so query.py can load them later
without re-computing anything.

Run this file to (re)build everything from scratch:

    python ingest.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


def load_documents(docs_dir: Path) -> dict:
    """Read every .md file in docs_dir into memory as {filename: full_text}."""
    documents = {}
    for path in sorted(docs_dir.glob("*.md")):
        documents[path.name] = path.read_text(encoding="utf-8")
    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split `text` into overlapping fixed-size windows of `chunk_size`
    characters, advancing by (chunk_size - overlap) each step.

    Why overlap matters: without it, a sentence sitting exactly on a chunk
    boundary gets torn in two — the tail of chunk N and the head of chunk
    N+1 each hold half a sentence. Neither half alone carries enough
    meaning for the embedding model to represent well, so retrieval can
    miss it even though the source document clearly answers the question.
    Overlap re-includes the last `overlap` characters of one chunk at the
    start of the next, so most sentences end up whole in at least one
    chunk somewhere.

    This is deliberately the simplest possible strategy: a fixed
    character-count sliding window that doesn't know about words,
    sentences, or markdown structure (headings, tables, lists). It's
    enough to see RAG's mechanics end to end. A more careful pipeline
    would chunk on paragraph/heading boundaries or use a token-aware
    splitter — worth revisiting once you've seen where naive chunking
    causes retrieval to miss things.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def build_chunks(docs_dir: Path) -> list:
    """
    Load every doc, chunk it, and attach metadata to each chunk.

    This metadata is what makes access control possible later: doc_type
    and employee_id travel with the chunk text all the way through
    embedding and retrieval, so query.py can filter chunks by who's asking
    without ever re-reading or re-parsing the source files.
    """
    documents = load_documents(docs_dir)
    all_chunks = []

    for filename, text in documents.items():
        # config.PERSONAL_DOCS maps filename -> owning employee_id for the
        # two sensitive docs; everything else is a general policy doc.
        employee_id = config.PERSONAL_DOCS.get(filename)  # None for general docs
        doc_type = "personal" if employee_id else "general"

        pieces = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "text": piece,
                    "source_file": filename,
                    "chunk_id": f"{filename}::chunk_{i}",
                    "doc_type": doc_type,
                    "employee_id": employee_id,
                }
            )

        print(f"{filename}: {len(text)} chars -> {len(pieces)} chunk(s)  [{doc_type}]")

    return all_chunks


def embed_chunks(chunks: list, model: SentenceTransformer) -> np.ndarray:
    """
    Turn each chunk's text into a fixed-length vector ("embedding") that
    represents its meaning: chunks about similar topics end up as vectors
    that sit close together in this 384-dimensional space, unrelated
    chunks end up far apart. That's the core trick RAG relies on —
    instead of keyword matching, we'll later compare *meaning* via
    vector distance.

    Why 384 dimensions specifically? That's just how all-MiniLM-L6-v2
    was trained/designed — every model has a fixed "output size" baked
    in. Crucially, the model always outputs exactly 384 numbers no
    matter how long the input text is: a 3-word chunk and a 500-
    character chunk both collapse down to one vector of the same
    length. That fixed size is exactly what FAISS requires next — every
    vector stored in a FAISS index must have identical dimensionality,
    because "distance between vectors" is only a meaningful comparison
    when every vector lives in the same-shaped space.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # FAISS requires float32 specifically (not float64/numpy's default in
    # some paths) — cast explicitly so the contract is obvious here rather
    # than relying on whatever dtype sentence-transformers happens to return.
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a "flat L2" FAISS index — the simplest index type FAISS offers.
    "Flat" means it stores every vector exactly as given, with no
    compression or approximation. "L2" means similarity is measured as
    literal Euclidean distance between vectors: smaller distance = more
    similar meaning.

    At search time, a flat index compares the query vector against
    *every* stored vector one by one (brute force, O(n) per query) and
    returns the closest ones. That's completely fine at our scale — 34
    vectors — and it's the easiest index type to reason about. Large-
    scale RAG systems (millions of vectors) use approximate index types
    like IVF or HNSW that trade a little accuracy for much faster
    search; not a concern here.
    """
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_index_and_metadata(index: faiss.Index, chunks: list) -> None:
    """
    Persist two files to disk, and critically, keep them in sync:

    1. faiss_index.bin — the vectors themselves, in insertion order.
       FAISS assigns each vector a position (0, 1, 2, ...) matching the
       order it was added; it does NOT store the original chunk text or
       metadata — just the raw numbers.

    2. chunks_metadata.json — a plain list where position i holds the
       text + metadata for the chunk whose vector lives at position i in
       the FAISS index.

    These two files are two halves of one system: FAISS can tell us
    "position 7 is the closest match to your query," but only this
    metadata list can turn "position 7" back into actual chunk text and
    its source_file/doc_type/employee_id. They must always be rebuilt
    together and stay the same length/order, which is exactly what
    running this whole script top-to-bottom guarantees.
    """
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    chunks = build_chunks(config.HR_DOCS_DIR)
    print(f"\nTotal chunks generated: {len(chunks)}")

    print(f"\nLoading embedding model '{config.EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Embedding chunks...")
    embeddings = embed_chunks(chunks, model)
    print(f"Embeddings array shape: {embeddings.shape}  (n_chunks x 384 dimensions)")

    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)
    print(f"FAISS index contains {index.ntotal} vectors of dimension {index.d}")

    save_index_and_metadata(index, chunks)
    print(f"\nSaved FAISS index to:      {config.FAISS_INDEX_PATH}")
    print(f"Saved chunk metadata to:  {config.METADATA_PATH}")
