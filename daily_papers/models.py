from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HFPaperEntry:
    paper_id: str
    title: str
    summary: str
    authors: List[str]
    upvotes: int
    published_at: str
    hf_url: Optional[str] = None
    arxiv_url: Optional[str] = None
    pdf_url: Optional[str] = None
    origin: Optional[str] = None


@dataclass
class ExtractedText:
    abstract: str = ""
    intro: str = ""
    conclusion: str = ""
    full_text: str = ""
