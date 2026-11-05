import json

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    DOCS_DIR,
    EMBEDDING_MODEL,
    INDEX_NAME,
    NAMESPACE,
    UPSERT_BATCH_SIZE,
    require_env,
)

CATEGORIES = {
    "pinecone-serverless.md": "infraestructura",
    "bm25-okapi.md": "recuperacion",
    "chunking.md": "ingesta",
    "embeddings-openai.md": "embeddings",
    "metricas-rag.md": "evaluacion",
}


def load_documents() -> list[Document]:
    paths = sorted(DOCS_DIR.glob("*.md"))
    if not paths:
        raise RuntimeError(f"No hay markdown en {DOCS_DIR}")
    documents = []
    for path in paths:
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={
                    "source": path.name,
                    "doc_id": path.stem,
                    "category": CATEGORIES.get(path.name, "general"),
                },
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    pages: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        page = pages.get(source, 0)
        pages[source] = page + 1
        chunk.metadata["page"] = page
        chunk.metadata["chunk_id"] = f"{chunk.metadata['doc_id']}#{page}"
    return chunks


def clear_namespace() -> None:
    """Vacía solo el namespace de trabajo. No toca el resto del índice."""
    index = Pinecone(api_key=require_env("PINECONE_API_KEY")).Index(INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
    except Exception as exc:
        message = str(exc).lower()
        if "namespace not found" in message or "not found" in message:
            return
        raise


def save_chunks(chunks: list[Document]) -> None:
    payload = [
        {
            "id": chunk.metadata["chunk_id"],
            "text": chunk.page_content,
            "metadata": {
                "source": chunk.metadata["source"],
                "doc_id": chunk.metadata["doc_id"],
                "category": chunk.metadata["category"],
                "page": chunk.metadata["page"],
                "chunk_id": chunk.metadata["chunk_id"],
            },
        }
        for chunk in chunks
    ]
    CHUNKS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_chunks(chunks: list[Document]) -> None:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=require_env("OPENAI_API_KEY"),
    )
    store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
        pinecone_api_key=require_env("PINECONE_API_KEY"),
        text_key="text",
    )
    store.add_texts(
        texts=[chunk.page_content for chunk in chunks],
        metadatas=[dict(chunk.metadata) for chunk in chunks],
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        namespace=NAMESPACE,
        batch_size=UPSERT_BATCH_SIZE,
        async_req=False,
    )


def ingest() -> int:
    chunks = split_documents(load_documents())
    if not chunks:
        raise RuntimeError("El splitter no produjo chunks")
    clear_namespace()
    upsert_chunks(chunks)
    save_chunks(chunks)
    print(
        f"Ingesta lista: {len(chunks)} chunks en {INDEX_NAME}/{NAMESPACE}. "
        f"Corpus BM25 en {CHUNKS_PATH.name}."
    )
    return len(chunks)


if __name__ == "__main__":
    ingest()
