"""Reranking for improved retrieval quality."""

from rank_bm25 import BM25Okapi
from loguru import logger

from .embeddings import EmbeddingModel


class Reranker:
    """Hybrid reranking using BM25 + semantic similarity."""

    def __init__(self, alpha: float = 0.5):
        """
        Initialize reranker.

        Args:
            alpha: Weight for semantic score (1-alpha for BM25).
        """
        self.alpha = alpha
        self.embedding_model = EmbeddingModel()

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank results using hybrid BM25 + semantic scoring.

        Args:
            query: The search query.
            results: List of search results with 'content' field.
            top_k: Number of results to return.

        Returns:
            Reranked list of results.
        """
        if not results:
            return []

        logger.debug(f"Reranking {len(results)} results")

        # Get documents
        documents = [r["content"] for r in results]

        # BM25 scoring
        bm25_scores = self._bm25_score(query, documents)

        # Semantic similarity scoring (use existing scores if available)
        if "relevance_score" in results[0]:
            semantic_scores = [r["relevance_score"] for r in results]
        else:
            semantic_scores = self._semantic_score(query, documents)

        # Normalize scores
        bm25_normalized = self._normalize(bm25_scores)
        semantic_normalized = self._normalize(semantic_scores)

        # Combine scores
        combined_scores = [
            self.alpha * sem + (1 - self.alpha) * bm25
            for sem, bm25 in zip(semantic_normalized, bm25_normalized)
        ]

        # Add scores to results and sort
        for i, result in enumerate(results):
            result["bm25_score"] = bm25_scores[i]
            result["semantic_score"] = semantic_scores[i]
            result["combined_score"] = combined_scores[i]

        reranked = sorted(results, key=lambda x: x["combined_score"], reverse=True)

        return reranked[:top_k]

    def _bm25_score(self, query: str, documents: list[str]) -> list[float]:
        """Calculate BM25 scores."""
        tokenized_docs = [doc.lower().split() for doc in documents]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)

        return scores.tolist()

    def _semantic_score(self, query: str, documents: list[str]) -> list[float]:
        """Calculate semantic similarity scores."""
        query_embedding = self.embedding_model.embed_text(query)
        doc_embeddings = self.embedding_model.embed_batch(documents)

        scores = [
            self.embedding_model.similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        return scores

    def _normalize(self, scores: list[float]) -> list[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]
