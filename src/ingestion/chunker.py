"""Intelligent chunking strategies for research papers."""

from dataclasses import dataclass, field
from loguru import logger

from .pdf_parser import ParsedPaper


@dataclass
class Chunk:
    """A chunk of text with metadata for retrieval."""

    content: str
    chunk_id: str
    page_numbers: list[int]
    section: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "page_numbers": self.page_numbers,
            "section": self.section,
            **self.metadata,
        }


class SemanticChunker:
    """Chunk documents with awareness of document structure."""

    # Common section headers in research papers
    SECTION_MARKERS = [
        "abstract",
        "introduction",
        "background",
        "related work",
        "methodology",
        "methods",
        "approach",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "references",
        "appendix",
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_paper(self, paper: ParsedPaper) -> list[Chunk]:
        """Chunk a parsed paper into retrievable segments."""
        logger.info(f"Chunking paper: {paper.metadata.title}")

        chunks = []

        # Process each page
        for page in paper.pages:
            page_chunks = self._chunk_page(
                text=page.text,
                page_number=page.page_number,
                paper_title=paper.metadata.title,
            )
            chunks.extend(page_chunks)

        logger.info(f"Created {len(chunks)} chunks from paper")
        return chunks

    def _chunk_page(
        self,
        text: str,
        page_number: int,
        paper_title: str,
    ) -> list[Chunk]:
        """Chunk a single page's text."""
        if not text.strip():
            return []

        chunks = []
        current_section = self._detect_section(text)

        # Split into sentences/paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this paragraph starts a new section
            detected = self._detect_section(para)
            if detected:
                current_section = detected

            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        content=current_chunk,
                        page_number=page_number,
                        section=current_section,
                        chunk_index=chunk_index,
                        paper_title=paper_title,
                    ))
                    chunk_index += 1

                    # Keep overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                content=current_chunk,
                page_number=page_number,
                section=current_section,
                chunk_index=chunk_index,
                paper_title=paper_title,
            ))

        return chunks

    def _detect_section(self, text: str) -> str:
        """Detect which section a piece of text belongs to."""
        text_lower = text.lower()[:100]  # Check beginning only

        for section in self.SECTION_MARKERS:
            if section in text_lower:
                return section

        return ""

    def _create_chunk(
        self,
        content: str,
        page_number: int,
        section: str,
        chunk_index: int,
        paper_title: str,
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        chunk_id = f"{paper_title[:30]}_{page_number}_{chunk_index}"

        return Chunk(
            content=content.strip(),
            chunk_id=chunk_id,
            page_numbers=[page_number],
            section=section,
            metadata={
                "paper_title": paper_title,
                "char_count": len(content),
            },
        )
