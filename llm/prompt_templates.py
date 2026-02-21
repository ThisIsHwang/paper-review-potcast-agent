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

from typing import List, Optional

from daily_papers.models import ExtractedText, HFPaperEntry

SYSTEM_PROMPT = (
    "You are an expert science communicator and presentation designer who transforms research papers "
    "into structured, engaging, YouTube-ready slide decks. "
    "Your narration style should be concise, vivid, and accessible to a broad technical audience, "
    "while maintaining factual precision."
)


def _figures_section(figure_summaries: Optional[List[str]]) -> str:
    if not figure_summaries:
        return "(No pre-extracted figures/tables available.)"
    lines = ["Available figures/tables (reference by number/type in figure_hint if useful):"]
    for item in figure_summaries:
        lines.append(f"- {item}")
    return "\n".join(lines)



def build_user_prompt(paper: HFPaperEntry, extracted: ExtractedText, figure_summaries: Optional[List[str]] = None) -> str:
    authors = ", ".join(paper.authors)

    prompt = f"""
Your task is to convert the following research paper into a clean, structured JSON representation
suitable for a YouTube slide-based presentation with voice-over narration.

Audience expectations:
- Clear, engaging scientific storytelling.
- Short, high-impact bullets.
- Narration that flows naturally and avoids dense academic phrasing.
- Accuracy is essential, but the tone should feel alive and clear.
- Do not omit major sections or key contributions; add slides if needed to cover the full story.

Paper metadata:
- id: {paper.paper_id}
- title: {paper.title}
- authors: {authors}
- summary: {paper.summary}
- arxiv: {paper.arxiv_url or "n/a"}
- pdf: {paper.pdf_url or "n/a"}

Pre-extracted figures/tables you MAY use:
{_figures_section(figure_summaries)}

Figure selection guidance:
- Use at most 5 figures/tables to support the story.
- Favor:
  • motivation diagrams  
  • architecture/framework/pipeline visuals  
  • summary tables illustrating results  
  • key findings/insights  
- If one figure/table is especially central, create an image-only slide by:
  - omitting bullets, AND
  - setting figure_hint to that specific asset.
- If no figure is appropriate, leave figure_hint empty.

You may insert an image-only spotlight slide anywhere if it improves narrative clarity.

Extracted text snippets:
Abstract:
{extracted.abstract}

Introduction:
{extracted.intro}

Conclusion:
{extracted.conclusion}

Full paper text:
{extracted.full_text}

### Required JSON Schema
Your output must strictly follow this structure:

{{
  "paper_id": "string",
  "title": "string",
  "category": "string",
  "origin": "institution/lab/company/university of the first author (short phrase)",
  "slides": [
    {{
      "title": "string",
      "bullets": ["string"],
      "script": "natural, engaging narration spoken over this slide",
      "figure_hint": "optional, e.g., 'Figure 4' or 'Table 2'"
    }}
  ]
}}

### Style & quality expectations:
- Bullets: short, skimmable, high-signal.
- Script: conversational, smooth, and suitable for spoken narration—not written academic prose.
- Do not include bracketed delivery cues or stage directions in scripts.
- Preserve technical accuracy at all times.
- The narrative should feel cohesive across slides.
- Ensure coverage of major sections (problem/motivation, method, results, limitations, and key takeaways).
- Identify the first author's institution concisely; infer if reasonable. Use "" only if truly unknown.

Return **only** the JSON object. No explanations.
"""
    return prompt
