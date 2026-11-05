import unittest

from src.metrics import precision_at_k, recall_at_k


class MetricsTest(unittest.TestCase):
    def test_recall_hits_when_expected_source_is_in_top_k(self):
        sources = ["otro", "pinecone-serverless.md", "bm25-okapi.md"]
        self.assertEqual(recall_at_k(sources, "pinecone-serverless.md", k=5), 1.0)

    def test_recall_misses_when_expected_source_is_absent(self):
        sources = ["bm25-okapi.md", "chunking.md"]
        self.assertEqual(recall_at_k(sources, "pinecone-serverless.md", k=5), 0.0)

    def test_precision_counts_only_useful_hits(self):
        sources = ["metricas-rag.md", "otro.md", "metricas-rag.md", "x.md", "y.md"]
        self.assertAlmostEqual(precision_at_k(sources, "metricas-rag.md", k=5), 2 / 5)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            precision_at_k(["a"], "a", k=0)


if __name__ == "__main__":
    unittest.main()
