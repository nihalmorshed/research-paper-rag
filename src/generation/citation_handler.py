"""Handle citation extraction and formatting."""

import re
from dataclasses import dataclass


@dataclass
class Citation:
    """A citation reference."""

    source_index: int
    page_numbers: list[int]
    section: str
    paper_title: str


class CitationHandler:
    """Extract and format citations from LLM responses."""

    def __init__(self):
        # Pattern to match [Source N] citations
        self.citation_pattern = re.compile(r"\[Source\s+(\d+)\]")

    def extract_citations(
        self,
        response: str,
        context_chunks: list[dict],
    ) -> tuple[str, list[Citation]]:
        """
        Extract citations from response and link to source chunks.

        Returns:
            Tuple of (response with enhanced citations, list of Citation objects)
        """
        citations = []
        cited_indices = set()

        # Find all citation references
        matches = self.citation_pattern.finditer(response)

        for match in matches:
            source_idx = int(match.group(1)) - 1  # Convert to 0-indexed

            if 0 <= source_idx < len(context_chunks):
                cited_indices.add(source_idx)

                chunk = context_chunks[source_idx]
                metadata = chunk.get("metadata", {})

                # Parse page numbers
                page_str = metadata.get("page_numbers", "[]")
                try:
                    pages = eval(page_str) if isinstance(page_str, str) else page_str
                except Exception:
                    pages = []

                citations.append(Citation(
                    source_index=source_idx + 1,
                    page_numbers=pages if isinstance(pages, list) else [pages],
                    section=metadata.get("section", ""),
                    paper_title=metadata.get("paper_title", ""),
                ))

        return response, citations

    def format_citations_footer(
        self,
        citations: list[Citation],
    ) -> str:
        """Format citations as a footer section."""
        if not citations:
            return ""

        lines = ["\n---\n**Sources:**"]

        # Deduplicate by source index
        seen = set()
        unique_citations = []
        for cit in citations:
            if cit.source_index not in seen:
                seen.add(cit.source_index)
                unique_citations.append(cit)

        for cit in unique_citations:
            page_info = f"Page(s) {', '.join(map(str, cit.page_numbers))}" if cit.page_numbers else ""
            section_info = f"Section: {cit.section}" if cit.section else ""

            parts = [f"[Source {cit.source_index}]"]
            if cit.paper_title:
                parts.append(f"*{cit.paper_title}*")
            if page_info:
                parts.append(page_info)
            if section_info:
                parts.append(section_info)

            lines.append("- " + " - ".join(parts))

        return "\n".join(lines)

    def enhance_response(
        self,
        response: str,
        context_chunks: list[dict],
    ) -> str:
        """Enhance response with detailed citation footer."""
        response, citations = self.extract_citations(response, context_chunks)
        footer = self.format_citations_footer(citations)

        return response + footer
