"""Embedding models for semantic search."""

from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np

from ..config import settings


class EmbeddingModel:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
