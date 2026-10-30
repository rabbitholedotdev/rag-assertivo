"""Environment-driven config. No hostnames, no personal collection names."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "publico_br")
COLLECTION_NAMES = [
    c.strip()
    for c in os.getenv("COLLECTION_NAMES", "").split(",")
    if c.strip()
] or [COLLECTION_NAME]

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
RERANK_MODEL = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
RAG_RERANK = os.getenv("RAG_RERANK", "1") not in ("0", "false", "no")
RAG_POOLS_SEPARADOS = os.getenv("RAG_POOLS_SEPARADOS", "0") not in ("0", "false", "no")
RAG_PENALIDADE_SINTETICO = float(os.getenv("RAG_PENALIDADE_SINTETICO", "0.0"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

RAG_GEN_MODEL = os.getenv("RAG_GEN_MODEL", "extractive")
RAG_VERIFY_MODEL = os.getenv("RAG_VERIFY_MODEL") or None
RAG_DEADLINE_S = float(os.getenv("RAG_DEADLINE_S", "60"))

MCP_HTTP_PATH = os.getenv("MCP_HTTP_PATH", "/mcp")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN") or None
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
