"""Tests for the ingestion module."""

import pytest
from src.ingestion.chunker import SemanticChunker, Chunk
from src.ingestion.pdf_parser import PaperMetadata, PageContent, ParsedPaper


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)

        metadata = PaperMetadata(
            title="Test Paper",
            authors=["Author One"],
            total_pages=1,
        )

        page = PageContent(
            page_number=1,
            text="This is a test paragraph.\n\nThis is another paragraph.",
        )

        paper = ParsedPaper(metadata=metadata, pages=[page])
        chunks = chunker.chunk_paper(paper)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_section_detection(self):
        """Test section header detection."""
        chunker = SemanticChunker()

        assert chunker._detect_section("Abstract\nThis paper presents...") == "abstract"
        assert chunker._detect_section("1. Introduction\nWe propose...") == "introduction"
        assert chunker._detect_section("Random text here") == ""


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_to_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(
            content="Test content",
            chunk_id="test_1",
            page_numbers=[1, 2],
            section="introduction",
            metadata={"paper_title": "Test"},
        )

        result = chunk.to_dict()

        assert result["content"] == "Test content"
        assert result["chunk_id"] == "test_1"
        assert result["page_numbers"] == [1, 2]
        assert result["section"] == "introduction"
        assert result["paper_title"] == "Test"
