# Copyright 2026 ThisIsHwang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import List

import requests

from .models import HFPaperEntry


def _parse_authors(raw) -> List[str]:
    authors: List[str] = []
    if not raw:
        return authors
    for a in raw:
        if isinstance(a, dict):
            authors.append(a.get("name") or a.get("user", "") or "")
        else:
            authors.append(str(a))
    return [a for a in authors if a]


def fetch_daily_papers(date: str, top_k: int, base_url: str = "https://huggingface.co") -> List[HFPaperEntry]:
    url = f"{base_url.rstrip('/')}/api/daily_papers"
    params = {"date": date}
    logging.info("Fetching Daily Papers for %s", date)
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logging.error("Failed to fetch Daily Papers: %s", exc)
        return []

    papers: List[HFPaperEntry] = []
    for item in data:
        paper_info = item.get("paper") or {}
        paper_id = paper_info.get("arxivId") or paper_info.get("id") or item.get("paper_id") or item.get("id") or ""
        title = paper_info.get("title") or item.get("title") or ""
        summary = paper_info.get("summary") or item.get("summary") or ""
        authors = _parse_authors(paper_info.get("authors") or item.get("authors") or [])
        upvotes = int(paper_info.get("upvotes") or item.get("upvotes") or 0)
        published_at = paper_info.get("publishedAt") or item.get("publishedAt") or item.get("published_at") or ""
        hf_url = paper_info.get("url") or item.get("url") or item.get("hf_url")
        arxiv_url = paper_info.get("arxivUrl") or item.get("arxiv_url")
        pdf_url = paper_info.get("pdfUrl") or item.get("pdf_url")

        try:
            papers.append(
                HFPaperEntry(
                    paper_id=str(paper_id),
                    title=title,
                    summary=summary,
                    authors=authors,
                    upvotes=upvotes,
                    published_at=published_at,
                    hf_url=hf_url,
                    arxiv_url=arxiv_url,
                    pdf_url=pdf_url,
                )
            )
        except Exception as exc:
            logging.warning("Skipping malformed entry: %s", exc)

    papers.sort(key=lambda p: p.upvotes, reverse=True)
    if top_k > 0:
        papers = papers[:top_k]
    logging.info("Fetched %d papers (top_k=%d)", len(papers), top_k)
    return papers


def resolve_arxiv_and_pdf(entry: HFPaperEntry) -> HFPaperEntry:
    """Fill arxiv and pdf URLs when missing. Simple fallback for MVP."""
    if not entry.arxiv_url and entry.paper_id:
        entry.arxiv_url = f"https://arxiv.org/abs/{entry.paper_id}"
    if not entry.pdf_url and entry.paper_id:
        entry.pdf_url = f"https://arxiv.org/pdf/{entry.paper_id}.pdf"
    return entry
