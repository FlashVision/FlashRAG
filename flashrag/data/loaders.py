"""
Document loaders for PDF, HTML, Markdown, CSV, and plain text.

Each loader reads a file and returns a list of ``Document`` dicts with keys
``text``, ``metadata`` (source path, page number, etc.).
"""

from __future__ import annotations

import csv
import html as html_mod
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Document(len={len(self.text)}, preview='{preview}…')"


class DocumentLoader:
    """Auto-detect file type and dispatch to the correct loader."""

    _loaders: dict[str, type] = {}

    @classmethod
    def register(cls, ext: str, loader_cls: type) -> None:
        cls._loaders[ext.lower().lstrip(".")] = loader_cls

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        if path.is_dir():
            return self._load_directory(path)
        return self._load_file(path)

    def _load_file(self, path: Path) -> list[Document]:
        ext = path.suffix.lower().lstrip(".")
        if ext in self._loaders:
            loader = self._loaders[ext]()
            return loader.load(path)
        return PlainTextLoader().load(path)

    def _load_directory(self, directory: Path) -> list[Document]:
        supported = {".pdf", ".html", ".htm", ".md", ".csv", ".txt"}
        docs: list[Document] = []
        for fpath in sorted(directory.rglob("*")):
            if fpath.is_file() and fpath.suffix.lower() in supported:
                docs.extend(self._load_file(fpath))
        return docs


class PlainTextLoader:
    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(text=text, metadata={"source": str(path), "type": "text"})]


class PDFLoader:
    """Load PDF files using PyPDF2 (with pdfplumber fallback for tables)."""

    def load(self, path: str | Path, use_pdfplumber: bool = False) -> list[Document]:
        path = Path(path)
        if use_pdfplumber:
            return self._load_pdfplumber(path)
        return self._load_pypdf2(path)

    def _load_pypdf2(self, path: Path) -> list[Document]:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF loading. Install with: pip install 'flashrag[pdf]'"
            )

        reader = PdfReader(str(path))
        docs: list[Document] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                docs.append(
                    Document(
                        text=text,
                        metadata={"source": str(path), "page": i + 1, "type": "pdf"},
                    )
                )
        return docs

    def _load_pdfplumber(self, path: Path) -> list[Document]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required. Install with: pip install 'flashrag[pdf]'")

        docs: list[Document] = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        cells = [str(c) if c else "" for c in row]
                        table_text += " | ".join(cells) + "\n"

                combined = text.strip()
                if table_text.strip():
                    combined += "\n\n[Table]\n" + table_text.strip()

                if combined.strip():
                    docs.append(
                        Document(
                            text=combined,
                            metadata={
                                "source": str(path),
                                "page": i + 1,
                                "type": "pdf",
                                "has_tables": bool(tables),
                            },
                        )
                    )
        return docs


class HTMLLoader:
    """Load HTML files, stripping tags and extracting visible text."""

    _TAG_RE = re.compile(r"<[^>]+>")
    _SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = self._html_to_text(raw)
        return [Document(text=text, metadata={"source": str(path), "type": "html"})]

    def _html_to_text(self, html: str) -> str:
        text = self._SCRIPT_RE.sub("", html)
        text = self._TAG_RE.sub(" ", text)
        text = html_mod.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class MarkdownLoader:
    """Load Markdown files with optional header-based section splitting."""

    def load(self, path: str | Path, split_on_headers: bool = False) -> list[Document]:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")

        if not split_on_headers:
            return [Document(text=text, metadata={"source": str(path), "type": "markdown"})]

        sections = self._split_headers(text, str(path))
        return (
            sections
            if sections
            else [Document(text=text, metadata={"source": str(path), "type": "markdown"})]
        )

    def _split_headers(self, text: str, source: str) -> list[Document]:
        header_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(header_re.finditer(text))
        if not matches:
            return []

        docs: list[Document] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                docs.append(
                    Document(
                        text=section_text,
                        metadata={
                            "source": source,
                            "type": "markdown",
                            "header": m.group(2),
                            "level": len(m.group(1)),
                        },
                    )
                )
        return docs


class CSVLoader:
    """Load CSV files, converting each row into a text document."""

    def load(
        self,
        path: str | Path,
        text_columns: list[str] | None = None,
        delimiter: str = ",",
    ) -> list[Document]:
        path = Path(path)
        docs: list[Document] = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row_idx, row in enumerate(reader):
                if text_columns:
                    parts = [str(row.get(c, "")) for c in text_columns if row.get(c)]
                else:
                    parts = [f"{k}: {v}" for k, v in row.items() if v]
                text = "\n".join(parts)
                if text.strip():
                    docs.append(
                        Document(
                            text=text,
                            metadata={"source": str(path), "type": "csv", "row": row_idx},
                        )
                    )
        return docs


DocumentLoader.register("pdf", PDFLoader)
DocumentLoader.register("html", HTMLLoader)
DocumentLoader.register("htm", HTMLLoader)
DocumentLoader.register("md", MarkdownLoader)
DocumentLoader.register("csv", CSVLoader)
DocumentLoader.register("txt", PlainTextLoader)
