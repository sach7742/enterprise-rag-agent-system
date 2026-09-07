import os
import pickle
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


class HybridRetriever:
    def __init__(self, qdrant_path: str = "qdrant_db", bm25_path: str = "bm25_index.pkl"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.qdrant_client = QdrantClient(path=qdrant_path)
        self.collection_name = "enterprise_docs"
        self.bm25_index_path = bm25_path

    def _get_query_embedding(self, query: str) -> List[float]:
        return self.embedding_model.embed_query(query)

    def _dense_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.qdrant_client.collection_exists(self.collection_name):
            return []

        query_vector = self._get_query_embedding(query)
        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        return [
            {
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": point.payload.get("metadata", {}),
            }
            for point in response.points
        ]

    def _sparse_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not os.path.exists(self.bm25_index_path):
            return []

        with open(self.bm25_index_path, "rb") as f:
            data = pickle.load(f)
            bm25 = data["index"]
            chunks = data["chunks"]

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

        return [
            {
                "text": chunks[i],
                "score": float(scores[i]),
                "metadata": {"chunk_id": i},
            }
            for i in top_indices if scores[i] > 0
        ]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        dense_results = self._dense_search(query, limit=top_k)
        sparse_results = self._sparse_search(query, limit=top_k)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        k = 60

        for rank, doc in enumerate(dense_results):
            text = doc["text"]
            doc_map[text] = doc
            rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (k + rank + 1))

        for rank, doc in enumerate(sparse_results):
            text = doc["text"]
            doc_map[text] = doc
            rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (k + rank + 1))

        sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[text] for text in sorted_docs[:top_k]]

    def close(self):
        """Explicitly release storage locks and close Qdrant connection."""
        if hasattr(self, "qdrant_client") and self.qdrant_client is not None:
            self.qdrant_client.close()