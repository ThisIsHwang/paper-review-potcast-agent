import logging
import re
from typing import Optional

# Prefer pymupdf import to avoid name collision with other "fitz" packages.
try:  # pragma: no cover - import guard
    import pymupdf as fitz  # type: ignore
except Exception:  # pragma: no cover - fallback
    import fitz  # type: ignore

from .models import ExtractedText


def _find_section(text: str, start_keywords: list[str], stop_keywords: list[str]) -> str:
    lowered = text.lower()
    start_idx: Optional[int] = None
    for key in start_keywords:
        idx = lowered.find(key.lower())
        if idx != -1:
            start_idx = idx
            break
    if start_idx is None:
        return ""

    stop_positions = []
    for key in stop_keywords:
        idx = lowered.find(key.lower(), start_idx + 1)
        if idx != -1:
            stop_positions.append(idx)
    stop_idx = min(stop_positions) if stop_positions else None
    snippet = text[start_idx:stop_idx] if stop_idx else text[start_idx:]
    # Trim header keyword
    snippet = re.sub(r"^(abstract|introduction|conclusion)s?:", "", snippet, flags=re.IGNORECASE).strip()
    return snippet.strip()


def extract_core_text(pdf_path: str) -> ExtractedText:
    logging.info("Extracting core text from %s", pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logging.error("Failed to open PDF %s: %s", pdf_path, exc)
        return ExtractedText()

    pages = []
    for i, page in enumerate(doc):
        try:
            pages.append(page.get_text("text"))
        except Exception as exc:
            logging.warning("Skipping page %d: %s", i, exc)
    joined = "\n".join(pages)

    abstract = _find_section(joined, ["abstract"], ["introduction", "background"])
    intro = _find_section(joined, ["introduction"], ["related work", "method", "methods", "approach"])
    conclusion = _find_section(joined, ["conclusion", "conclusions"], ["references", "acknowledgments", "acknowledgements"])
    return ExtractedText(
        abstract=abstract,
        intro=intro,
        conclusion=conclusion,
        full_text=joined,
    )


def count_pages(pdf_path: str) -> int:
    """Return page count for the given PDF path (0 on failure)."""
    try:
        doc = fitz.open(pdf_path)
        return len(doc)
    except Exception as exc:
        logging.warning("Failed to count pages for %s: %s", pdf_path, exc)
        return 0
