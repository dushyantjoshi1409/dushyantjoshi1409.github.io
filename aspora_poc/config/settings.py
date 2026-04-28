"""
Centralized configuration — all env vars and model settings in one place.
No hardcoded values scattered across the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# --- LLM Configuration ---
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS: int = 1024
LLM_TEMPERATURE_FACTUAL: float = 0.0      # deterministic for financial data
LLM_TEMPERATURE_GENERAL: float = 0.3      # slightly creative for general knowledge

# --- Embedding Configuration ---
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# --- Retrieval Configuration ---
RRF_K: int = 60                            # standard RRF parameter
RETRIEVAL_TOP_K: int = 5                   # default number of docs to retrieve
RERANKER_TOP_K: int = 3                    # final docs after reranking
BM25_TOP_K: int = 10                       # BM25 candidates before fusion
FAISS_TOP_K: int = 10                      # FAISS candidates before fusion

# --- Langfuse Tracing ---
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED: bool = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# --- Server Configuration ---
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# --- Database Paths ---
SQLITE_DB_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stocks.db")
FAISS_INDEX_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index.bin")

# --- Logging ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
