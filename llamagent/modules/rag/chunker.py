"""
Document chunking for the RAG module.

Routes documents to format-specific chunkers based on file extension:
- MarkdownChunker: splits by headers, then paragraphs within sections
- CodeChunker: splits by top-level definitions (functions, classes)
- PlainTextChunker: splits by paragraphs, merging short ones

Each chunker produces Chunk objects with text and metadata (source, chunk_index,
and optionally section_title or language).
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of text with associated metadata."""
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata keys: source (filename), chunk_index (int),
    # section_title (str, optional), language (str, optional)


# ======================================================================
# Extension-to-language mapping for CodeChunker
# ======================================================================

_EXT_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
}

# Extensions recognized as code files
CODE_EXTENSIONS = set(_EXT_LANGUAGE_MAP.keys())


# ======================================================================
# DocumentChunker — top-level router
# ======================================================================


class DocumentChunker:
    """Routes documents to format-specific chunkers based on file extension."""

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def chunk(self, content: str, filepath: str) -> list[Chunk]:
        """Split document content into chunks based on file type."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".md":
            return MarkdownChunker(self.chunk_size).chunk(content, filepath)
        elif ext in CODE_EXTENSIONS:
            return CodeChunker(self.chunk_size).chunk(content, filepath)
        else:
            return PlainTextChunker(self.chunk_size).chunk(content, filepath)


# ======================================================================
# MarkdownChunker
# ======================================================================


class MarkdownChunker:
    """
    Split Markdown documents by headers (# / ## / ###).

    Each section becomes a chunk. If a section exceeds chunk_size,
    split by paragraphs within it. Preserves section title in metadata.
    """

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def chunk(self, content: str, filepath: str) -> list[Chunk]:
        """Split Markdown content into chunks."""
        filename = os.path.basename(filepath)
        sections = self._split_by_headers(content)
        chunks: list[Chunk] = []

        for title, body in sections:
            section_text = f"{title}\n{body}".strip() if title else body.strip()
            if not section_text:
                continue

            if len(section_text) <= self.chunk_size:
                chunks.append(Chunk(
                    text=section_text,
                    metadata={
                        "source": filename,
                        "chunk_index": len(chunks),
                        "section_title": title or "",
                    },
                ))
            else:
                # Split oversized section by paragraphs
                sub_chunks = self._split_by_paragraphs(section_text)
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        text=sc,
                        metadata={
                            "source": filename,
                            "chunk_index": len(chunks),
                            "section_title": title or "",
                        },
                    ))

        return chunks

    @staticmethod
    def _split_by_headers(content: str) -> list[tuple[str, str]]:
        """
        Split Markdown content by header lines.

        Returns a list of (title, body) tuples. The first section may have
        an empty title if the document doesn't start with a header.
        """
        # Match lines starting with 1-6 # characters followed by a space
        header_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        sections: list[tuple[str, str]] = []
        last_end = 0
        last_title = ""

        for match in header_pattern.finditer(content):
            # Collect body text between previous header and this one
            body = content[last_end:match.start()]
            if last_end == 0 and body.strip():
                # Content before the first header
                sections.append(("", body.strip()))
            elif last_end > 0:
                sections.append((last_title, body.strip()))

            last_title = match.group(1).strip()
            last_end = match.end()

        # Remaining content after the last header
        remaining = content[last_end:].strip()
        if last_title or remaining:
            sections.append((last_title, remaining))

        # If no headers found, return entire content as one section
        if not sections:
            sections.append(("", content.strip()))

        return sections

    def _split_by_paragraphs(self, text: str) -> list[str]:
        """Split text by paragraphs, merging short ones up to chunk_size."""
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 <= self.chunk_size:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > self.chunk_size:
                    for j in range(0, len(para), self.chunk_size):
                        chunks.append(para[j:j + self.chunk_size])
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks


# ======================================================================
# CodeChunker
# ======================================================================


class CodeChunker:
    """
    Split source code by top-level definitions.

    Language-specific heuristics:
    - Python: lines starting with 'def ' or 'class ' at indent level 0
    - JS/TS: 'function ', 'class ', 'const ... = ', 'export '
    - Other languages: fall back to blank-line separation

    If a definition block exceeds chunk_size, force-split by lines.
    Stores language in metadata.
    """

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def chunk(self, content: str, filepath: str) -> list[Chunk]:
        """Split source code into chunks."""
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        language = _EXT_LANGUAGE_MAP.get(ext, "unknown")

        if language == "python":
            blocks = self._split_python(content)
        elif language in ("javascript", "typescript"):
            blocks = self._split_js_ts(content)
        else:
            blocks = self._split_by_blank_lines(content)

        chunks: list[Chunk] = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if len(block) <= self.chunk_size:
                chunks.append(Chunk(
                    text=block,
                    metadata={
                        "source": filename,
                        "chunk_index": len(chunks),
                        "language": language,
                    },
                ))
            else:
                # Force-split oversized blocks by lines
                sub_chunks = self._force_split_lines(block)
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        text=sc,
                        metadata={
                            "source": filename,
                            "chunk_index": len(chunks),
                            "language": language,
                        },
                    ))

        return chunks

    @staticmethod
    def _split_python(content: str) -> list[str]:
        """Split Python code by top-level def/class definitions."""
        # Match lines at indent level 0 starting with def or class
        pattern = re.compile(r"^(?=(?:def |class )\S)", re.MULTILINE)
        return CodeChunker._split_by_pattern(content, pattern)

    @staticmethod
    def _split_js_ts(content: str) -> list[str]:
        """Split JS/TS code by top-level definitions."""
        # Match lines at indent level 0 starting with function, class, const, export
        pattern = re.compile(
            r"^(?=(?:function |class |const \w+ ?= |export ))",
            re.MULTILINE,
        )
        return CodeChunker._split_by_pattern(content, pattern)

    @staticmethod
    def _split_by_pattern(content: str, pattern: re.Pattern) -> list[str]:
        """Split content by a regex pattern matching definition boundaries."""
        positions = [m.start() for m in pattern.finditer(content)]
        if not positions:
            return [content] if content.strip() else []

        blocks: list[str] = []
        # Content before the first match (e.g., imports, module-level comments)
        if positions[0] > 0:
            preamble = content[:positions[0]].strip()
            if preamble:
                blocks.append(preamble)

        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(content)
            block = content[pos:end].rstrip()
            if block:
                blocks.append(block)

        return blocks

    @staticmethod
    def _split_by_blank_lines(content: str) -> list[str]:
        """Fallback: split by blank lines for languages without specific heuristics."""
        blocks = re.split(r"\n\s*\n", content)
        return [b.strip() for b in blocks if b.strip()]

    def _force_split_lines(self, block: str) -> list[str]:
        """Force-split a large block by lines up to chunk_size."""
        lines = block.split("\n")
        chunks: list[str] = []
        current = ""

        for line in lines:
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single line exceeds chunk_size, include it as-is
                if len(line) > self.chunk_size:
                    chunks.append(line)
                    current = ""
                else:
                    current = line

        if current:
            chunks.append(current)

        return chunks


# ======================================================================
# PlainTextChunker
# ======================================================================


class PlainTextChunker:
    """
    Split plain text by paragraphs (double newlines).

    Merges short paragraphs up to chunk_size. Force-splits oversized
    paragraphs. This is an improved version of the original _split_text logic.
    """

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def chunk(self, content: str, filepath: str) -> list[Chunk]:
        """Split plain text content into chunks."""
        filename = os.path.basename(filepath)
        raw_chunks = self._split_paragraphs(content)
        chunks: list[Chunk] = []

        for text in raw_chunks:
            chunks.append(Chunk(
                text=text,
                metadata={
                    "source": filename,
                    "chunk_index": len(chunks),
                },
            ))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Split text by double newlines (paragraphs).

        Strategy: paragraph boundaries -> merge short paragraphs up to
        chunk_size -> force-split oversized paragraphs at chunk_size.
        """
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 <= self.chunk_size:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current)
                if len(para) > self.chunk_size:
                    for j in range(0, len(para), self.chunk_size):
                        chunks.append(para[j:j + self.chunk_size])
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks
