"""
document_loader.py — Ingest financial documents from PDF, plain text, or HTML.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from models import DocumentInput, DocumentType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document-type detection heuristics
# ---------------------------------------------------------------------------

_TYPE_PATTERNS: list[tuple[DocumentType, list[str]]] = [
    (DocumentType.EARNINGS_RELEASE, [
        r"earnings release", r"quarterly results", r"q[1-4]\s+f?y?\d{4}",
        r"fiscal.*quarter", r"quarterly.*revenue",
    ]),
    (DocumentType.TEN_K, [r"\b10-k\b", r"annual report on form 10-k", r"annual report"]),
    (DocumentType.TEN_Q, [r"\b10-q\b", r"quarterly report on form 10-q"]),
    (DocumentType.PRESS_RELEASE, [r"press release", r"for immediate release", r"contact:"]),
]


def detect_document_type(text: str) -> DocumentType:
    """Return the most likely document type based on keyword heuristics."""
    sample = text[:3000].lower()
    for doc_type, patterns in _TYPE_PATTERNS:
        if any(re.search(p, sample) for p in patterns):
            return doc_type
    return DocumentType.UNKNOWN


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------

def _load_pdf_pdfplumber(path: str) -> tuple[str, int]:
    """Extract text from a PDF using pdfplumber (preferred)."""
    import pdfplumber  # type: ignore

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                pages.append(extracted)
    return "\n\n".join(pages), page_count


def _load_pdf_pymupdf(path: str) -> tuple[str, int]:
    """Extract text from a PDF using PyMuPDF (fallback)."""
    import fitz  # type: ignore  # PyMuPDF

    doc = fitz.open(path)
    page_count = len(doc)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages), page_count


def _load_pdf(path: str) -> tuple[str, int]:
    """Try pdfplumber, fall back to PyMuPDF."""
    try:
        text, pages = _load_pdf_pdfplumber(path)
        logger.debug("Loaded PDF with pdfplumber (%d pages)", pages)
        return text, pages
    except Exception as exc:
        logger.warning("pdfplumber failed (%s); trying PyMuPDF", exc)

    try:
        text, pages = _load_pdf_pymupdf(path)
        logger.debug("Loaded PDF with PyMuPDF (%d pages)", pages)
        return text, pages
    except Exception as exc:
        raise RuntimeError(f"Could not extract text from PDF '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# HTML loading
# ---------------------------------------------------------------------------

def _load_html(path: str) -> tuple[str, None]:
    """Extract visible text from an HTML file using BeautifulSoup."""
    from bs4 import BeautifulSoup  # type: ignore

    with open(path, encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")
    for tag in soup(["script", "style", "meta", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return text, None


# ---------------------------------------------------------------------------
# Plain-text loading
# ---------------------------------------------------------------------------

def _load_text(path: str) -> tuple[str, None]:
    """Read a plain-text file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read(), None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(path: str) -> DocumentInput:
    """Load a financial document from *path* and return a :class:`DocumentInput`.

    Supported formats: ``.pdf``, ``.txt``, ``.html`` / ``.htm``.
    Document type is auto-detected from content heuristics.

    Args:
        path: Absolute or relative path to the document file.

    Returns:
        :class:`DocumentInput` with extracted text and metadata.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If text extraction fails for all available backends.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = resolved.suffix.lower()

    if ext == ".pdf":
        raw_text, page_count = _load_pdf(str(resolved))
    elif ext in {".html", ".htm"}:
        raw_text, page_count = _load_html(str(resolved))
    elif ext in {".txt", ".md", ""}:
        raw_text, page_count = _load_text(str(resolved))
    else:
        # Try as plain text for unknown extensions
        logger.warning("Unknown extension '%s'; attempting plain-text load.", ext)
        raw_text, page_count = _load_text(str(resolved))

    detected_type = detect_document_type(raw_text)
    logger.info(
        "Loaded '%s' — type=%s, pages=%s, chars=%d",
        resolved.name,
        detected_type.value,
        page_count,
        len(raw_text),
    )

    return DocumentInput(
        raw_text=raw_text,
        source_file=str(resolved),
        detected_type=detected_type,
        page_count=page_count,
    )
