
"""Configuration for the DAT401 evidence-grounded research assistant."""

from pathlib import Path
import os


# ==================================================
# PROJECT DIRECTORIES
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        str(PROJECT_DIR / "data")
    )
)

ARTIFACT_DIR = Path(
    os.getenv(
        "ARTIFACT_DIR",
        str(PROJECT_DIR / "artifacts")
    )
)

CHUNKS_FILE = Path(
    os.getenv(
        "CHUNKS_FILE",
        str(DATA_DIR / "section_aware_chunks.jsonl")
    )
)


# ==================================================
# SAVED RETRIEVAL INDEXES
# ==================================================

BM25_FILE = ARTIFACT_DIR / "bm25.pkl"

FAISS_FILE = ARTIFACT_DIR / "dense.faiss"

CHUNKS_PICKLE_FILE = ARTIFACT_DIR / "chunks.pkl"

INDEX_METADATA_FILE = ARTIFACT_DIR / "index_metadata.json"


# ==================================================
# RETRIEVAL MODELS
# ==================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CROSS_ENCODER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


# ==================================================
# OPENROUTER LANGUAGE MODEL
# ==================================================

# Dynamic free-model routing through OpenRouter.
# The actual selected model is recorded in generation logs.

LLM_MODEL_NAME = "openrouter/free"

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)

# Maximum requested generation tokens.
# The provider may impose a lower limit.

MAX_NEW_TOKENS = 1800

# Deterministic generation for consistent research answers.
LLM_TEMPERATURE = 0.0

# Timeout settings for OpenRouter requests.
# These values are used by generation.py.
OPENROUTER_CONNECT_TIMEOUT = 20

OPENROUTER_READ_TIMEOUT = 180


# ==================================================
# RETRIEVAL SETTINGS
# ==================================================

# BGE query instruction.
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

# Reciprocal Rank Fusion constant.
RRF_K = 60

# Initial retrieval candidate counts.
BM25_CANDIDATES = 20

DENSE_CANDIDATES = 20

# Number of hybrid candidates passed to Cross-Encoder.
HYBRID_CANDIDATES_FOR_RERANKING = 40

# Number of unique article passages supplied to the LLM.
FINAL_TOP_K = 5


# ==================================================
# RELEVANCE FILTERING
# ==================================================

# Leave relevance threshold disabled until score
# calibration has been completed.
#
# Cross-Encoder scores are not probabilities.
# Do not select an arbitrary threshold.

RERANK_SCORE_THRESHOLD = None

# Maximum number of passages retained per article.
MAX_PASSAGES_PER_ARTICLE = 1


# ==================================================
# HUGGING FACE CACHE
# ==================================================

HF_HOME = os.getenv(
    "HF_HOME",
    str(PROJECT_DIR / ".hf_cache")
)
