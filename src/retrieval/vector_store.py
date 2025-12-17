"""Vector store using ChromaDB for document storage and retrieval."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from loguru import logger

from ..config import settings
from ..ingestion.chunker import Chunk
from .embeddings import EmbeddingModel


class VectorStore:
    """ChromaDB-based vector store for document chunks."""

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str | None = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir

        # Initialize ChromaDB
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_model = EmbeddingModel()
        logger.info(f"Initialized vector store: {collection_name}")

    def add_chunks(self, chunks: list[Chunk], paper_id: str) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} chunks for paper: {paper_id}")

        # Prepare data for ChromaDB
        ids = [f"{paper_id}_{chunk.chunk_id}" for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "paper_id": paper_id,
                "page_numbers": str(chunk.page_numbers),
                "section": chunk.section,
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            for chunk in chunks
        ]

        # Generate embeddings
        embeddings = self.embedding_model.embed_batch(documents)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Successfully added {len(chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int | None = None,
        paper_id: str | None = None,
    ) -> list[dict]:
        """Search for relevant chunks."""
        top_k = top_k or settings.top_k

        # Build where clause for filtering
        where = {"paper_id": paper_id} if paper_id else None

        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })

        return formatted

    def delete_paper(self, paper_id: str) -> None:
        """Delete all chunks for a specific paper."""
        self.collection.delete(where={"paper_id": paper_id})
        logger.info(f"Deleted chunks for paper: {paper_id}")

    def list_papers(self) -> list[str]:
        """List all paper IDs in the store."""
        results = self.collection.get(include=["metadatas"])

        paper_ids = set()
        for metadata in results["metadatas"]:
            if "paper_id" in metadata:
                paper_ids.add(metadata["paper_id"])

        return list(paper_ids)

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "papers": len(self.list_papers()),
        }
