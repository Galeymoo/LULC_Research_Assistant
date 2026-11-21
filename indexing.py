"""Build a BM25 index and a FAISS dense index from the chunk JSONL corpus.

The same ordered chunk list is saved alongside both indexes so retrieval IDs,
article metadata and page numbers remain aligned.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    ARTIFACT_DIR,
    BM25_FILE,
    CHUNKS_FILE,
    CHUNKS_PICKLE_FILE,
    DENSE_CANDIDATES,
    EMBEDDING_MODEL_NAME,
    FAISS_FILE,
    INDEX_METADATA_FILE,
)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25.

    IMPORTANT: Compare this with the tokenizer used in Notebook 07. If the
    notebook used a different tokenizer (e.g. NLTK word_tokenize), change this
    function to match it, then rebuild the indexes for comparable results.
    """
    return re.findall(r"(?u)\b\w+\b", str(text).lower())


def load_chunk_records(chunks_file: Path) -> list[dict[str, Any]]:
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {chunks_file}\n"
            "Copy section_aware_chunks.jsonl into the project's data/ folder."
        )
    frame = pd.read_json(chunks_file, lines=True)
    required = {"chunk_id", "article_id", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Corpus is missing required fields: {sorted(missing)}")

    frame["text"] = frame["text"].fillna("").astype(str)
    frame = frame[frame["text"].str.strip().ne("")].copy()
    if frame["chunk_id"].duplicated().any():
        duplicates = frame.loc[frame["chunk_id"].duplicated(), "chunk_id"].head().tolist()
        raise ValueError(f"Duplicate chunk_id values found, examples: {duplicates}")

    # Convert numpy/pandas values to plain Python values for reliable pickling/UI.
    records = json.loads(frame.to_json(orient="records", force_ascii=False))
    return records


def build_indexes() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_chunk_records(CHUNKS_FILE)
    texts = [str(row["text"]) for row in records]

    print(f"Loaded {len(records):,} non-empty chunks.")
    print("Building BM25 index...")
    tokenized_corpus = [tokenize_for_bm25(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 and its ordered records. Position i in BM25 corresponds to
    # records[i], and FAISS vector i also corresponds to records[i].
    with BM25_FILE.open("wb") as handle:
        pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with CHUNKS_PICKLE_FILE.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Encoding passages. This is a one-time build step and may take a while.")
    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("Embedding dimensions do not match the chunk count.")

    # Inner-product search over unit-normalized vectors is cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_FILE))

    metadata = {
        "chunk_count": len(records),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "faiss_index_type": "IndexFlatIP with normalized embeddings",
        "bm25_tokenizer": "unicode word regex: (?u)\\b\\w+\\b, lowercase",
        "candidate_counts": {
            "bm25": 20,
            "dense": DENSE_CANDIDATES,
        },
        "warning": (
            "For exact comparability with Notebook 07, ensure the BM25 tokenizer "
            "and corpus ordering match Notebook 07."
        ),
    }
    INDEX_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nIndex build complete:")
    print(f"  BM25: {BM25_FILE}")
    print(f"  FAISS: {FAISS_FILE}")
    print(f"  Chunk metadata: {CHUNKS_PICKLE_FILE}")
    print(f"  Metadata: {INDEX_METADATA_FILE}")
    print("Index positions in BM25 and FAISS are aligned to chunks.pkl.")


if __name__ == "__main__":
    build_indexes()
