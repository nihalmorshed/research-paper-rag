"""PDF parsing with text, table, and figure extraction."""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""

    page_number: int
    text: str
    tables: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


@dataclass
class PaperMetadata:
    """Metadata extracted from a research paper."""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    filename: str = ""
    total_pages: int = 0


@dataclass
class ParsedPaper:
    """Complete parsed research paper."""

    metadata: PaperMetadata
    pages: list[PageContent]

    def get_full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.text for page in self.pages)


class PDFParser:
    """Extract text, tables, and figures from PDF research papers."""

    def __init__(self):
        self.supported_extensions = {".pdf"}

    def parse(self, file_path: str | Path) -> ParsedPaper:
        """Parse a PDF file and extract all content."""
        file_path = Path(file_path)

        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        logger.info(f"Parsing PDF: {file_path.name}")

        doc = fitz.open(file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_content = self._extract_page_content(page, page_num + 1)
            pages.append(page_content)

        metadata = self._extract_metadata(doc, file_path.name)
        doc.close()

        logger.info(f"Extracted {len(pages)} pages from {file_path.name}")

        return ParsedPaper(metadata=metadata, pages=pages)

    def _extract_page_content(self, page: fitz.Page, page_number: int) -> PageContent:
        """Extract content from a single page."""
        text = page.get_text("text")

        # Extract images/figures
        images = []
        for img_index, img in enumerate(page.get_images()):
            images.append({
                "index": img_index,
                "page": page_number,
            })

        return PageContent(
            page_number=page_number,
            text=text.strip(),
            images=images,
        )

    def _extract_metadata(self, doc: fitz.Document, filename: str) -> PaperMetadata:
        """Extract paper metadata."""
        metadata = doc.metadata

        # Try to extract title from metadata or first page
        title = metadata.get("title", "")
        if not title and len(doc) > 0:
            first_page_text = doc[0].get_text("text")
            lines = first_page_text.split("\n")
            title = lines[0] if lines else ""

        # Extract authors if available
        authors = []
        if metadata.get("author"):
            authors = [a.strip() for a in metadata["author"].split(",")]

        return PaperMetadata(
            title=title,
            authors=authors,
            filename=filename,
            total_pages=len(doc),
        )

    def extract_abstract(self, paper: ParsedPaper) -> str:
        """Attempt to extract abstract from paper."""
        full_text = paper.get_full_text().lower()

        # Look for abstract section
        abstract_start = full_text.find("abstract")
        if abstract_start == -1:
            return ""

        # Find end of abstract (usually "introduction" or "1.")
        text_after_abstract = full_text[abstract_start + 8:]
        intro_pos = text_after_abstract.find("introduction")
        section_pos = text_after_abstract.find("1.")

        end_pos = min(
            pos for pos in [intro_pos, section_pos, 2000]
            if pos > 0
        )

        abstract = text_after_abstract[:end_pos].strip()
        return abstract
