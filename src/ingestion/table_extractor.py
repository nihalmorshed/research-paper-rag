"""Extract tables from PDF documents."""

import pdfplumber
from pathlib import Path
from dataclasses import dataclass
from loguru import logger


@dataclass
class ExtractedTable:
    """A table extracted from a PDF."""

    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers and not self.rows:
            return ""

        lines = []

        # Headers
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Rows
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    def to_text(self) -> str:
        """Convert table to plain text representation."""
        lines = []

        if self.headers:
            lines.append("\t".join(self.headers))

        for row in self.rows:
            lines.append("\t".join(str(cell) for cell in row))

        return "\n".join(lines)


class TableExtractor:
    """Extract tables from PDF documents using pdfplumber."""

    def extract_tables(self, file_path: str | Path) -> list[ExtractedTable]:
        """Extract all tables from a PDF file."""
        file_path = Path(file_path)
        tables = []

        logger.info(f"Extracting tables from: {file_path.name}")

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()

                for table_idx, table_data in enumerate(page_tables):
                    if not table_data:
                        continue

                    extracted = self._process_table(table_data, page_num, table_idx)
                    if extracted:
                        tables.append(extracted)

        logger.info(f"Extracted {len(tables)} tables from {file_path.name}")
        return tables

    def _process_table(
        self,
        table_data: list[list],
        page_number: int,
        table_index: int
    ) -> ExtractedTable | None:
        """Process raw table data into structured format."""
        if not table_data or len(table_data) < 1:
            return None

        # Clean cells
        cleaned = []
        for row in table_data:
            cleaned_row = [
                str(cell).strip() if cell else ""
                for cell in row
            ]
            cleaned.append(cleaned_row)

        # Assume first row is header
        headers = cleaned[0] if cleaned else []
        rows = cleaned[1:] if len(cleaned) > 1 else []

        return ExtractedTable(
            page_number=page_number,
            table_index=table_index,
            headers=headers,
            rows=rows,
        )
