def precision_at_k(retrieved_sources: list[str], expected_source: str, k: int = 5) -> float:
    """Fracción de los k recuperados que vienen del documento esperado."""
    if k < 1:
        raise ValueError("k debe ser >= 1")
    top = retrieved_sources[:k]
    useful = sum(1 for source in top if source == expected_source)
    return useful / k


def recall_at_k(retrieved_sources: list[str], expected_source: str, k: int = 5) -> float:
    """1 si el documento esperado aparece entre los k recuperados."""
    if k < 1:
        raise ValueError("k debe ser >= 1")
    return 1.0 if expected_source in retrieved_sources[:k] else 0.0
