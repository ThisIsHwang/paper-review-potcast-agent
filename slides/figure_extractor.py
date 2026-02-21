"""Extract figures/tables and captions from a PDF into captions.json.

Intended to mirror the ad-hoc test script logic and drop assets next to each
downloaded paper so the slide pipeline can auto-embed figures.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Optional heavy deps are imported lazily inside functions.


_LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
_CAPTION_PREFER = {
    "Figure": re.compile(r"^(fig(?:ure)?\.?\s*\d+(?:\.\d+)*)", re.IGNORECASE),
    "Table": re.compile(r"^(table\s*\d+(?:\.\d+)*)", re.IGNORECASE),
}
_NUMBER_PATTERNS = [
    r"\bFigure\.?\s*(\d+(?:\.\d+)*)",
    r"\bFig\.?\s*(\d+(?:\.\d+)*)",
    r"\bTable\.?\s*(\d+(?:\.\d+)*)",
]


@dataclass
class _ExtractedItem:
    page: int
    block_type: str
    detected_idx: int
    number: Optional[str]
    caption: str
    file: str


def _overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    min_width = max(1e-6, min(a_end - a_start, b_end - b_start))
    return overlap / min_width


def _extract_caption(block_type: str, bbox: Tuple[float, float, float, float], text_blocks, search_margin=350, min_overlap=0.2) -> str:
    x1, y1, x2, y2 = bbox
    below_candidates = []
    above_candidates = []
    prefer_pattern = _CAPTION_PREFER.get(block_type)

    def register_candidate(distance, overlap, text, target):
        target.append((distance, -overlap, text))

    for tb in text_blocks:
        tx1, ty1, tx2, ty2, text, *rest = tb
        text = (text or "").strip()
        if not text:
            continue
        if len(rest) >= 2 and rest[1] != 0:
            continue

        overlap = _overlap_ratio(x1, x2, tx1, tx2)
        if overlap < min_overlap:
            continue

        below_dist = ty1 - y2
        above_dist = y1 - ty2
        has_prefix = bool(prefer_pattern.search(text)) if prefer_pattern else False

        if 0 <= below_dist <= search_margin:
            register_candidate((0 if has_prefix else 1, below_dist), overlap, text, below_candidates)
        if 0 <= above_dist <= search_margin:
            register_candidate((0 if has_prefix else 1, above_dist), overlap, text, above_candidates)

    primary, fallback = (below_candidates, above_candidates) if block_type == "Figure" else (above_candidates, below_candidates)

    def pick(cands):
        prefix = [c for c in cands if c[0][0] == 0]
        chosen_pool = prefix
        if not chosen_pool:
            return None
        chosen_pool.sort()
        return chosen_pool[0][2]

    for candidates in (primary, fallback):
        caption = pick(candidates)
        if caption:
            return caption

    return ""


def _extract_number(block_type: str, caption_text: str) -> Optional[str]:
    for pat in _NUMBER_PATTERNS:
        m = re.search(pat, caption_text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _ensure_model(config_path: Path, weights_path: Path):
    try:
        import layoutparser as lp
    except Exception as exc:  # pragma: no cover - optional dependency guard
        logging.warning("layoutparser missing; figure extraction skipped: %s", exc)
        return None

    extra_config = ["MODEL.WEIGHTS", str(weights_path), "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7]
    try:
        return lp.Detectron2LayoutModel(str(config_path), label_map=_LABEL_MAP, extra_config=extra_config)
    except Exception as exc:  # pragma: no cover - model load guard
        logging.warning("Failed to load PubLayNet model: %s", exc)
        return None


def _default_model_paths() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    base = root / "models" / "publaynet"
    return base / "config.yaml", base / "model_final.pth"


def _load_pdf(path: Path):
    try:  # pragma: no cover - import guard
        import pymupdf as fitz
    except Exception:  # pragma: no cover
        import fitz

    try:
        return fitz.open(path)
    except Exception as exc:  # pragma: no cover - open guard
        logging.warning("Failed to open PDF %s: %s", path, exc)
        return None


def _to_image(page):
    import numpy as np
    import cv2

    pix = page.get_pixmap(dpi=300)
    if pix.n == 4:
        mode = cv2.COLOR_BGRA2RGB
    elif pix.n == 3:
        mode = cv2.COLOR_BGR2RGB
    else:
        import pymupdf as fitz  # type: ignore

        pix = fitz.Pixmap(fitz.csRGB, pix)
        mode = cv2.COLOR_BGR2RGB

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, mode)


def _extract_items(doc, model, assets_dir: Path) -> List[_ExtractedItem]:
    import cv2

    scale = 300 / 72
    items: List[_ExtractedItem] = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        img = _to_image(page)
        layout = model.detect(img)

        text_blocks = []
        for tb in page.get_textpage().extractBLOCKS():
            x0, y0, x1, y1, text, block_no, block_type = tb
            text_blocks.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale, text, block_no, block_type))

        for i, block in enumerate(layout):
            if block.type not in ["Figure", "Table"]:
                continue

            x1, y1, x2, y2 = block.coordinates
            crop = img[int(y1) : int(y2), int(x1) : int(x2)]
            filename = f"page{page_idx + 1}_{block.type}_{i + 1}.png"
            assets_dir.mkdir(parents=True, exist_ok=True)
            dest = assets_dir / filename
            cv2.imwrite(str(dest), crop)

            caption = _extract_caption(block.type, (x1, y1, x2, y2), text_blocks)
            number = _extract_number(block.type, caption)
            items.append(
                _ExtractedItem(
                    page=page_idx + 1,
                    block_type=block.type,
                    detected_idx=i + 1,
                    number=number,
                    caption=caption,
                    file=str(Path(assets_dir.name) / filename),
                )
            )

    return items


def extract_pdf_figures(pdf_path: str, out_dir: Optional[str] = None) -> Optional[str]:
    """Run layout detection and write captions.json next to the PDF.

    Returns the path to captions.json if extraction succeeded, else None.
    Skips work if captions.json already exists.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logging.warning("PDF not found for figure extraction: %s", pdf_path)
        return None

    out_dir_path = Path(out_dir) if out_dir else pdf_path.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)
    captions_path = out_dir_path / "captions.json"
    assets_dir = out_dir_path / "figures"
    if captions_path.exists():
        logging.info("captions.json already exists, skipping extraction: %s", captions_path)
        return str(captions_path)

    default_cfg, default_weights = _default_model_paths()
    config_path = Path(os.getenv("PUBLayNET_CONFIG", default_cfg))
    weights_path = Path(os.getenv("PUBLayNET_WEIGHTS", default_weights))

    if not config_path.exists() or not weights_path.exists():
        logging.warning("PubLayNet model files missing (config: %s, weights: %s); skipping figure extraction", config_path, weights_path)
        return None

    model = _ensure_model(config_path, weights_path)
    if not model:
        return None

    doc = _load_pdf(pdf_path)
    if not doc:
        return None

    try:
        items = _extract_items(doc, model, assets_dir)
    except Exception as exc:  # pragma: no cover - runtime guard
        logging.warning("Figure extraction failed for %s: %s", pdf_path, exc)
        return None

    results = []
    for item in items:
        data = item.__dict__.copy()
        # Save a normalized `type` field so downstream consumers don't have to
        # special-case the internal block_type name.
        data.setdefault("type", item.block_type)
        results.append(data)
    try:
        captions_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        logging.info("Figure captions written to %s (%d items)", captions_path, len(results))
        return str(captions_path)
    except Exception as exc:  # pragma: no cover - IO guard
        logging.warning("Failed to write captions.json for %s: %s", pdf_path, exc)
        return None
