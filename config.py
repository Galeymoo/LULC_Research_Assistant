
"""Configuration for the DAT401 evidence-grounded research assistant."""

from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent

# Data and index directories
DATA_DIR = Path(
    os.getenv("DATA_DIR", str(PROJECT_DIR / "data"))
)

ARTIFACT_DIR = Path(
    os.getenv("ARTIFACT_DIR", str(PROJECT_DIR / "artifacts"))
)

CHUNKS_FILE = Path(
    os.getenv(
        "CHUNKS_FILE",
        str(DATA_DIR / "section_aware_chunks.jsonl")
    )
)

# Saved retrieval indexes
BM25_FILE = ARTIFACT_DIR / "bm25.pkl"
FAISS_FILE = ARTIFACT_DIR / "dense.faiss"
CHUNKS_PICKLE_FILE = ARTIFACT_DIR / "chunks.pkl"
INDEX_METADATA_FILE = ARTIFACT_DIR / "index_metadata.json"

# Retrieval models
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Hosted language model through OpenRouter
LLM_MODEL_NAME = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_NEW_TOKENS = 1200
LLM_TEMPERATURE = 0.0

# Retrieval settings
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

RRF_K = 60
BM25_CANDIDATES = 20
DENSE_CANDIDATES = 20
HYBRID_CANDIDATES_FOR_RERANKING = 20
FINAL_TOP_K = 5

# Hugging Face cache
HF_HOME = os.getenv(
    "HF_HOME",
    str(PROJECT_DIR / ".hf_cache")
)
