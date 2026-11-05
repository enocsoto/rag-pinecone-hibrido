# Preentrega 4: RAG híbrido con Pinecone

Servicio en Python que crea un índice serverless en Pinecone, parte documentación técnica en chunks, la embebe con `text-embedding-3-small` (1536 dimensiones) y responde con un recuperador híbrido: BM25 + similitud vectorial. `evaluate.py` mide Recall@5 y Precision@5 sobre un golden set de 5 preguntas.

## Qué hay

- `src/setup_index.py`: crea el índice si no existe (`aws`, `us-east-1`, cosine, dimensión 1536) y espera a que quede listo.
- `src/ingest.py`: lee `data/docs/*.md`, corta con `RecursiveCharacterTextSplitter` (2400 caracteres, overlap 300, cerca de 500–800 tokens), guarda el texto en la metadata y hace upsert en el namespace `tecnico`.
- `src/rag.py`: clase `RAGSystem`. Combina `BM25Retriever` y Pinecone con `EnsembleRetriever` (pesos 0.5 / 0.5) y devuelve 5 documentos.
- `evaluate.py`: Precision@5 y Recall@5. Un chunk es útil si su `source` es el `documento_id_esperado`. Recall@5 es 1 cuando ese archivo aparece al menos una vez.
- `data/golden_set.json`: 5 preguntas, una por documento.

El namespace aísla este corpus. Una reingesta borra solo `tecnico`. Las claves se leen de `.env` y no se commitean.

## Replicar el índice

Requisitos: Python 3.12, cuenta de Pinecone (el plan gratuito alcanza) y una API key de OpenAI.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Completar `.env`:

```text
PINECONE_API_KEY=...
OPENAI_API_KEY=...
INDEX_NAME=rag-preentrega-4
PINECONE_NAMESPACE=tecnico
```

`INDEX_NAME` tiene que ser nuevo, o un índice ya creado con dimensión 1536. Si existe con 512 o 768, `setup_index` corta.

```bash
python -m src.setup_index
python -m src.ingest
python evaluate.py
```

La ingesta escribe `data/chunks.json` (no se versiona). BM25 lee ese archivo, así que hay que ingerir en la misma máquina antes de evaluar. Las estadísticas de Pinecone pueden tardar unos segundos en reflejar el conteo; la búsqueda no depende de ese conteo.

Salida de la evaluación: cada pregunta, los `source` del top 5, Recall@5, Precision@5, y el promedio de las cinco.

Prueba local de las métricas, sin API:

```bash
python -m unittest tests/test_metrics.py
```
