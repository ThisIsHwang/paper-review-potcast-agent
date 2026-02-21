"""Helpers for wiring extracted figure/table images into slides."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_REF_RE = re.compile(r"(?i)\b(fig(?:ure)?|table)\s*\.?\s*(\d+(?:\.\d+)*)")


def _normalize_type(name: str) -> str:
    return "table" if name.lower().startswith("table") else "figure"


def extract_reference(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """Return (type, number) pair from text like "Figure 2" or "Table 1"."""
    if not text:
        return None
    match = _REF_RE.search(text)
    if not match:
        return None
    return _normalize_type(match.group(1)), match.group(2)


@dataclass
class FigureAsset:
    path: Path
    caption: str
    number: Optional[str]
    asset_type: str
    page: Optional[int] = None


class FigureLibrary:
    def __init__(self, assets: Iterable[FigureAsset]):
        self.assets: List[FigureAsset] = list(assets)
        self._by_key: Dict[Tuple[str, str], FigureAsset] = {}
        for asset in self.assets:
            if not asset.number:
                continue
            key = (_normalize_type(asset.asset_type), str(asset.number))
            # keep the first occurrence per key
            self._by_key.setdefault(key, asset)

    def find(self, ref_type: Optional[str], number: Optional[str]) -> Optional[FigureAsset]:
        if not (ref_type and number):
            return None
        return self._by_key.get((_normalize_type(ref_type), str(number)))

    def search_caption(self, needle: str) -> Optional[FigureAsset]:
        if not needle:
            return None
        needle = needle.lower()
        for asset in self.assets:
            if asset.caption and needle in asset.caption.lower():
                return asset
        return None


def rewrite_caption(asset: FigureAsset, fallback: Optional[str] = None, max_len: int = 180) -> Optional[str]:
    text = (asset.caption or fallback or "").strip()
    if not text:
        return None

    text = re.sub(r"^(figure|fig\.)\s*\d+(?:\.\d+)*[:.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(table)\s*\d+(?:\.\d+)*[:.]?\s*", "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())

    parts = re.split(r"(?<=[.!?])\s+", text)
    summary = parts[0] if parts else text
    if len(summary) > max_len:
        summary = summary[: max_len - 3].rsplit(" ", 1)[0] + "..."

    prefix = f"{asset.asset_type} {asset.number}" if asset.number else asset.asset_type
    return f"{prefix}: {summary}" if summary else prefix


def summarize_assets(library: FigureLibrary, limit: int = 20) -> List[str]:
    """Return up to `limit` figure/table descriptions using the ORIGINAL caption text.

    Captions are only whitespace-collapsed for readability; no truncation/rewrites.
    """

    items = []
    sorted_assets = sorted(
        library.assets,
        key=lambda a: (a.page or 0, int(a.number) if a.number and str(a.number).isdigit() else 999),
    )
    for asset in sorted_assets:
        caption = (asset.caption or "").strip()
        caption = " ".join(caption.split()) if caption else "(no caption)"
        label = f"{asset.asset_type} {asset.number}" if asset.number else asset.asset_type
        page = f" (p{asset.page})" if asset.page else ""
        items.append(f"{label}{page}: {caption}")
        if len(items) >= limit:
            break
    return items


def load_figure_library(captions_path: str) -> Optional[FigureLibrary]:
    path = Path(captions_path)
    if not path.exists():
        logging.info("Figure captions file not found: %s", captions_path)
        return None

    try:
        raw = json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Failed to read figure captions at %s: %s", captions_path, exc)
        return None

    base_dir = path.parent
    assets: List[FigureAsset] = []
    seen_keys = set()
    for item in raw:
        filename = item.get("file")
        if not filename:
            continue

        asset_type = item.get("type") or item.get("block_type") or "Figure"
        number = item.get("number")
        normalized_key = None
        if number is not None:
            normalized_key = (_normalize_type(str(asset_type)), str(number))
            if normalized_key in seen_keys:
                logging.debug("Skipping duplicate %s %s at %s", normalized_key[0], normalized_key[1], filename)
                continue

        asset_path = base_dir / filename
        if not asset_path.exists():
            logging.debug("Skipping missing figure asset: %s", asset_path)
            continue

        assets.append(
            FigureAsset(
                path=asset_path,
                caption=item.get("caption", ""),
                number=str(number) if number is not None else None,
                asset_type=str(asset_type),
                page=item.get("page"),
            )
        )
        if normalized_key:
            seen_keys.add(normalized_key)

    if not assets:
        logging.info("No usable figure assets found in %s", captions_path)
        return None

    logging.info("Loaded %d figure assets from %s", len(assets), captions_path)
    return FigureLibrary(assets)
