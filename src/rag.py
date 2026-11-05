import json

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    INDEX_NAME,
    NAMESPACE,
    TOP_K,
    require_env,
)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def load_corpus() -> list[Document]:
    if not CHUNKS_PATH.exists():
        raise RuntimeError("Falta data/chunks.json. Corré primero python -m src.ingest")
    records = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("data/chunks.json está vacío. Volvé a correr la ingesta.")
    return [
        Document(page_content=record["text"], metadata=record["metadata"], id=record["id"])
        for record in records
    ]


class RAGSystem:
    """Ensemble de BM25 local y similitud en Pinecone. Devuelve 5 documentos."""

    def __init__(self) -> None:
        documents = load_corpus()
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=require_env("OPENAI_API_KEY"),
        )
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=NAMESPACE,
            pinecone_api_key=require_env("PINECONE_API_KEY"),
            text_key="text",
        )
        lexical = BM25Retriever.from_documents(
            documents,
            preprocess_func=tokenize,
            bm25_params={"k1": 1.5, "b": 0.75},
            k=TOP_K,
        )
        semantic = vector_store.as_retriever(
            search_kwargs={"k": TOP_K, "namespace": NAMESPACE}
        )
        self.retriever = EnsembleRetriever(
            retrievers=[lexical, semantic],
            weights=[0.5, 0.5],
            id_key="chunk_id",
        )

    def search(self, query: str) -> list[Document]:
        if not query.strip():
            raise ValueError("La consulta está vacía")
        return list(self.retriever.invoke(query))[:TOP_K]
