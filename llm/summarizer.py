from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from daily_papers.models import ExtractedText, HFPaperEntry
from .client import LLMClient
from .prompt_templates import SYSTEM_PROMPT, build_user_prompt


@dataclass
class SlideSpec:
    title: str
    bullets: List[str]
    script: str
    figure_hint: Optional[str] = None
    figure_image: Optional[str] = None
    figure_caption: Optional[str] = None


@dataclass
class PaperSummary:
    paper_id: str
    title: str
    category: str
    one_line: str
    origin: str
    authors: List[str]
    key_ideas: List[str]
    insights: List[dict]
    slides: List[SlideSpec]


@dataclass
class DailyEpisode:
    date: str
    papers: List[PaperSummary]


def _parse_slides(slides_raw: List[Dict[str, Any]]) -> List[SlideSpec]:
    parsed: List[SlideSpec] = []
    for slide in slides_raw:
        parsed.append(
            SlideSpec(
                title=slide.get("title", "Slide"),
                bullets=slide.get("bullets") or [],
                script=slide.get("script", ""),
                figure_hint=slide.get("figure_hint"),
            )
        )
    return parsed


def summarize_paper(
    paper: HFPaperEntry,
    text: ExtractedText,
    llm_client: LLMClient,
    figure_summaries: Optional[List[str]] = None,
) -> PaperSummary:
    prompt = build_user_prompt(paper, text, figure_summaries)
    raw = llm_client.generate_json(SYSTEM_PROMPT, prompt)
    slides = _parse_slides(raw.get("slides", []))

    return PaperSummary(
        paper_id=raw.get("paper_id", paper.paper_id),
        title=raw.get("title", paper.title),
        category=raw.get("category", "AI"),
        one_line=raw.get("one_line", paper.summary[:150]),
        origin=raw.get("origin", "") or (paper.origin or ""),
        authors=paper.authors,
        key_ideas=raw.get("key_ideas", []),
        insights=raw.get("insights", []),
        slides=slides,
    )
