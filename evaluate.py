import json

from src.config import GOLDEN_SET_PATH, TOP_K
from src.metrics import precision_at_k, recall_at_k
from src.rag import RAGSystem


def load_golden_set() -> list[dict]:
    rows = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    if len(rows) < 5:
        raise RuntimeError("El golden set necesita al menos 5 preguntas")
    return rows


def evaluate() -> None:
    system = RAGSystem()
    rows = load_golden_set()
    recalls: list[float] = []
    precisions: list[float] = []

    for row in rows:
        question = row["pregunta"]
        expected = row["documento_id_esperado"]
        found = system.search(question)
        sources = [doc.metadata.get("source", "") for doc in found]
        recall = recall_at_k(sources, expected, k=TOP_K)
        precision = precision_at_k(sources, expected, k=TOP_K)
        recalls.append(recall)
        precisions.append(precision)
        print(f"Pregunta: {question}")
        print(f"Esperado: {expected}")
        print(f"Recuperados: {sources}")
        print(f"Recall@{TOP_K}: {recall:.1f}")
        print(f"Precision@{TOP_K}: {precision:.1f}")
        print("-" * 40)

    print(f"Promedio Recall@{TOP_K}: {sum(recalls) / len(recalls):.2f}")
    print(f"Promedio Precision@{TOP_K}: {sum(precisions) / len(precisions):.2f}")


if __name__ == "__main__":
    evaluate()
