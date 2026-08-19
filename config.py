import os

from dotenv import load_dotenv


# =========================
# Environment
# =========================

load_dotenv()


# =========================
# Ollama
# =========================

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434"
)


# =========================
# Models
# =========================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:8b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "qwen3-embedding:0.6b"
)


# =========================
# ChromaDB
# =========================

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "law-rag/chroma_db"
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "civil_law_v2"
)


# =========================
# Retrieval
# =========================

SEMANTIC_TOP_K = int(
    os.getenv(
        "SEMANTIC_TOP_K",
        "30"
    )
)

KEYWORD_TOP_K = int(
    os.getenv(
        "KEYWORD_TOP_K",
        "30"
    )
)

FINAL_TOP_K = int(
    os.getenv(
        "FINAL_TOP_K",
        "20"
    )
)

CONTEXT_TOP_K = int(
    os.getenv(
        "CONTEXT_TOP_K",
        "6"
    )
)


# =========================
# Answerability
# =========================

MAX_SEMANTIC_DISTANCE = float(
    os.getenv(
        "MAX_SEMANTIC_DISTANCE",
        "1.08"
    )
)


# =========================
# Generation
# =========================

MAX_GENERATION_ATTEMPTS = int(
    os.getenv(
        "MAX_GENERATION_ATTEMPTS",
        "2"
    )
)


# =========================
# API Security
# =========================

API_KEY = os.getenv(
    "API_KEY",
    ""
)
