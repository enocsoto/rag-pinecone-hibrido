import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data" / "docs"
CHUNKS_PATH = ROOT / "data" / "chunks.json"
GOLDEN_SET_PATH = ROOT / "data" / "golden_set.json"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHUNK_SIZE = 2400
CHUNK_OVERLAP = 300
UPSERT_BATCH_SIZE = 50
TOP_K = 5
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "tecnico").strip() or "tecnico"
INDEX_NAME = os.getenv("INDEX_NAME", "rag-preentrega-4").strip() or "rag-preentrega-4"


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta {name}. Copiá .env.example a .env y completalo.")
    return value
