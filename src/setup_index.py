import asyncio

from pinecone import Pinecone, ServerlessSpec

from src.config import EMBEDDING_DIMENSION, INDEX_NAME, require_env


def _ready(description) -> bool:
    status = description.status
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(status.ready)


async def setup_index() -> None:
    """Crea el índice serverless si no existe y espera a que quede listo."""
    api_key = require_env("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    while not _ready(pc.describe_index(INDEX_NAME)):
        await asyncio.sleep(1)

    description = pc.describe_index(INDEX_NAME)
    dimension = getattr(description, "dimension", None)
    if dimension not in (None, EMBEDDING_DIMENSION):
        raise RuntimeError(
            f"El índice {INDEX_NAME} tiene dimensión {dimension}. "
            "Hace falta 1536, la de text-embedding-3-small. "
            "Usá otro INDEX_NAME o borrá el índice viejo."
        )
    print(
        f"Índice listo: {INDEX_NAME} "
        f"(dimensión {EMBEDDING_DIMENSION}, cosine, aws/us-east-1)"
    )


if __name__ == "__main__":
    asyncio.run(setup_index())
