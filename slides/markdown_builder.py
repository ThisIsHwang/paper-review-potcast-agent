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
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llm.summarizer import DailyEpisode, PaperSummary, SlideSpec
from slides.figure_assets import FigureAsset, FigureLibrary, extract_reference, rewrite_caption


def _clean_origin(text: str) -> str:
    return text.strip().strip(' "“”') if text else ""


def _add_slide(
    lines: List[str],
    scripts: List[str],
    body: str,
    script: str,
) -> None:
    lines.append(body.strip())
    lines.append("")
    lines.append("---")
    scripts.append(script.strip())


def _inline_markdown_to_html(text: str) -> str:
    """
    마크다운 스타일링 개선:
    **bold** -> 단순히 굵게가 아니라 포인트 컬러를 적용한 strong 태그
    *italic* -> em 태그
    """
    # bold (색상 강조를 위해 class 추가 가능, 여기선 CSS에서 strong 태그 자체를 스타일링함)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text


def _list_to_html(bullets: List[str]) -> str:
    if not bullets:
        return ""
    items = "".join(f"<li>{_inline_markdown_to_html(b)}</li>" for b in bullets)
    return f"<ul>{items}</ul>"


def _text_slide_body(
    title: str,
    bullets: List[str],
    prefix: Optional[str] = None,
) -> str:
    full_title = f"{prefix} {title}" if prefix else title
    # 헤더에 그라데이션 라인 효과 등을 주기 위해 구조 유지
    header = (
        '<div class="slide-header">'
        f"<h1>{_inline_markdown_to_html(full_title)}</h1>"
        "</div>"
    )
    list_html = _list_to_html(bullets)
    content = (
        '<div class="slide-content">'
        f'<div class="text text-only">{list_html}</div>'
        "</div>"
    )
    return f'<div class="slide">{header}{content}</div>'


def _resolve_figure_asset(
    slide: SlideSpec,
    figure_library: Optional[FigureLibrary],
) -> Optional[FigureAsset]:
    if not figure_library:
        return None

    if slide.figure_hint:
        ref = extract_reference(slide.figure_hint)
        if ref:
            asset = figure_library.find(*ref)
            if asset:
                return asset

        asset = figure_library.search_caption(slide.figure_hint)
        if asset:
            return asset
    return None


def _attach_figure_to_slide(
    slide: SlideSpec,
    asset: Optional[FigureAsset],
    out_dir: Path,
    paper_idx: int,
    slide_idx: int,
) -> None:
    if not asset:
        return

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    suffix = asset.path.suffix or ".png"
    name_parts = [f"paper{paper_idx}", f"slide{slide_idx}"]
    if asset.number:
        name_parts.append(f"no{asset.number}")
    dest = figures_dir / ("_".join(name_parts) + suffix)

    try:
        shutil.copyfile(asset.path, dest)
    except Exception as exc:
        logging.warning("Failed to copy figure asset %s -> %s: %s", asset.path, dest, exc)
        return

    slide.figure_image = dest.resolve().as_posix()
    slide.figure_caption = rewrite_caption(asset, slide.figure_hint)


def _llm_slide_body(idx: int, slide_idx: int, slide: SlideSpec, single_pdf_mode: bool) -> str:
    bullets_html = ""
    if slide.bullets:
        items = "".join(
            f"<li>{_inline_markdown_to_html(b)}</li>" for b in slide.bullets
        )
        bullets_html = f"<ul>{items}</ul>"

    label = f"{slide_idx}" if single_pdf_mode else f"{idx}.{slide_idx}"
    # 숫자 라벨을 작고 세련되게 표시하기 위해 span으로 감쌈
    header_html = f'<span class="slide-number">{label}</span> {_inline_markdown_to_html(slide.title)}'
    header = f'<div class="slide-header"><h1>{header_html}</h1></div>'
    
    content = ""

    if slide.figure_image:
        caption = slide.figure_caption or slide.figure_hint or ""
        caption_html = _inline_markdown_to_html(caption) if caption else ""
        alt = caption or f"Slide {idx}.{slide_idx} figure"

        if not slide.bullets:
            # Full figure slide
            figure_block = (
                '<div class="figure-only">'
                f'<figure class="media featured">'
                f'<div class="img-wrapper"><img src="{slide.figure_image}" alt="{alt}" /></div>'
            )
            if caption_html:
                figure_block += f"<figcaption>{caption_html}</figcaption>"
            figure_block += "</figure></div>"
            content = figure_block
        else:
            # Split layout
            figure_block = (
                '<div class="split">'
                f'<div class="text">{bullets_html}</div>'
                '<figure class="media">'
                f'<div class="img-wrapper"><img src="{slide.figure_image}" alt="{alt}" /></div>'
            )
            if caption_html:
                figure_block += f"<figcaption>{caption_html}</figcaption>"
            figure_block += "</figure></div>"
            content = figure_block
    else:
        # Text only
        content = f'<div class="text text-only">{bullets_html}</div>'

    return f'<div class="slide">{header}<div class="slide-content">{content}</div></div>'


def build_daily_markdown(
    daily: DailyEpisode,
    out_path: str,
    figure_libraries: Optional[Dict[str, FigureLibrary]] = None,
    single_pdf_mode: bool = False,
) -> List[str]:
    """
    Build Marp markdown with enhanced modern CSS styling.
    """
    out_dir = Path(out_path).parent
    
    # --- Modern CSS Design Definition ---
    css_lines = [
        "  :root {",
        "    --primary-color: #2563eb;",       # Royal Blue
        "    --secondary-color: #64748b;",     # Slate Gray
        "    --accent-color: #3b82f6;",        # Lighter Blue
        "    --text-color: #1f2937;",          # Dark Gray (not pure black)
        "    --bg-color: #ffffff;",
        "    --code-bg: #f1f5f9;",
        "    --slide-max-width: 1120px;",
        "    --slide-font-size: 1.1em;",       # 폰트 사이즈 약간 키움
        "    --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);",
        "    --radius-md: 12px;",
        "  }",
        "",
        "  section {",
        "    background-color: var(--bg-color);",
        "    color: var(--text-color);",
        "    position: relative;",
        "    padding: 40px 60px;",             # 여백 조정
        "    box-sizing: border-box;",
        "    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;",
        "    letter-spacing: -0.01em;",
        "  }",
        "",
        "  /* --- Header Styling --- */",
        "  .slide-header {",
        "    position: sticky;",
        "    top: 0;",
        "    background: rgba(255, 255, 255, 0.95);", # 살짝 투명하게
        "    padding-bottom: 20px;",
        "    z-index: 10;",
        "    border-bottom: 3px solid var(--primary-color);", # 단순 선 대신 포인트 컬러 바
        "    margin-bottom: 24px;",
        "  }",
        "",
        "  .slide-header h1 {",
        "    margin: 0;",
        "    font-size: 1.5em;",
        "    font-weight: 700;",
        "    color: var(--primary-color);",
        "    line-height: 1.2;",
        "    display: flex;",
        "    align-items: center;",
        "    gap: 12px;",
        "  }",
        "",
        "  .slide-number {",
        "    background: var(--primary-color);",
        "    color: white;",
        "    font-size: 0.6em;",
        "    padding: 2px 8px;",
        "    border-radius: 6px;",
        "    vertical-align: middle;",
        "    font-weight: 600;",
        "  }",
        "",
        "  /* --- Layout & Content --- */",
        "  .slide {",
        "    display: flex;",
        "    flex-direction: column;",
        "    height: 100%;",
        "  }",
        "",
        "  .slide-content {",
        "    flex: 1;",
        "    display: flex;",
        "    flex-direction: column;",
        "    justify-content: center;", # 내용 수직 중앙 정렬 기본
        "  }",
        "",
        "  /* --- Typography --- */",
        "  strong {",
        "    color: var(--primary-color);",
        "    font-weight: 700;",
        "  }",
        "",
        "  em {",
        "    color: var(--secondary-color);",
        "    font-style: italic;",
        "  }",
        "",
        "  /* --- Lists --- */",
        "  .text ul {",
        "    padding-left: 0;",
        "    margin: 0;",
        "    list-style: none;", # 기본 불릿 제거
        "  }",
        "",
        "  .text li {",
        "    position: relative;",
        "    padding-left: 24px;",
        "    margin-bottom: 14px;",
        "    line-height: 1.6;",
        "    font-size: 1.0em;",
        "  }",
        "",
        "  .text li::before {", # 커스텀 불릿
        "    content: '•';",
        "    color: var(--primary-color);",
        "    font-weight: bold;",
        "    position: absolute;",
        "    left: 0;",
        "    font-size: 1.2em;",
        "    line-height: 1.5;",
        "  }",
        "",
        "  .text-only {",
        "    font-size: 1.1em;",
        "    max-height: 75vh;",
        "    overflow-y: auto;",
        "  }",
        "",
        "  /* --- Images & Media --- */",
        "  .media {",
        "    display: flex;",
        "    flex-direction: column;",
        "    align-items: center;",
        "    width: 100%;",
        "  }",
        "",
        "  .img-wrapper {",
        "    background: white;",
        "    padding: 8px;",
        "    border-radius: var(--radius-md);",
        "    box-shadow: var(--shadow-soft);",
        "    border: 1px solid #e5e7eb;",
        "    display: inline-block;",
        "  }",
        "",
        "  .media img {",
        "    max-width: 100%;",
        "    max-height: 60vh;",
        "    object-fit: contain;",
        "    border-radius: 6px;", # wrapper 내부 이미지도 살짝 둥글게
        "    display: block;",
        "  }",
        "",
        "  .media figcaption {",
        "    font-size: 0.85em;",
        "    color: var(--secondary-color);",
        "    margin-top: 12px;",
        "    text-align: center;",
        "    font-weight: 500;",
        "    background: #f8fafc;",
        "    padding: 4px 12px;",
        "    border-radius: 20px;",
        "    display: inline-block;",
        "  }",
        "",
        "  /* --- Featured (Image Only) --- */",
        "  .figure-only {",
        "    height: 100%;",
        "    display: flex;",
        "    align-items: center;",
        "    justify-content: center;",
        "  }",
        "",
        "  .figure-only .img-wrapper {",
        "    padding: 12px;",
        "    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);",
        "  }",
        "",
        "  .media.featured img {",
        "    max-height: 70vh;",
        "  }",
        "",
        "  /* --- Split Layout --- */",
        "  .split {",
        "    display: flex;",
        "    gap: 32px;",
        "    align-items: center;", # 수직 중앙 정렬
        "    height: 100%;",
        "  }",
        "",
        "  .split .text {",
        "    flex: 1;",
        "    font-size: 0.95em;",
        "  }",
        "",
        "  .split figure {",
        "    flex: 1.2;", # 이미지를 조금 더 넓게
        "    margin: 0;",
        "  }",
        "",
        "  .figure-hint {",
        "    margin-top: 16px;",
        "    font-size: 0.85em;",
        "    color: #d97706;", # Amber color for hints
        "    background: #fffbeb;",
        "    padding: 8px 12px;",
        "    border-radius: 6px;",
        "    border-left: 4px solid #d97706;",
        "  }",
        "",
        "  /* --- Responsive --- */",
        "  @media (max-width: 900px) {",
        "    .split { flex-direction: column; }",
        "    .slide-header { position: relative; }",
        "  }",
    ]

    lines: List[str] = [
        "---",
        "marp: true",
        "paginate: true",
        f'title: "Daily AI Papers - {daily.date}"',
        "style: |",
    ] + css_lines + [
        "---",
        "",
    ]
    
    scripts: List[str] = []

    # Intro slide (HTML로)
    if single_pdf_mode and daily.papers:
        paper = daily.papers[0]
        origin = _clean_origin(paper.origin)
        bullets = [
            f"**Title**: {paper.title}",
            "NLP 코기",
        ]
        if origin:
            bullets.insert(1, f"**Affiliation**: {origin}")
        
        # 인트로 슬라이드용 별도 제목 (더 크게)
        intro_body = _text_slide_body("꼬리의 꼬리를 무는<br>페이퍼 딥다이브", bullets)
        origin_phrase = f"{origin}에서 나온 " if origin else ""
        intro_script = (
            f"안녕하세요. NLP 코기입니다. "
            f"오늘의 꼬리의 꼬리를 무는 페이퍼 딥다이브, 꼬꼬페에서 다룰 내용은 {origin_phrase}{paper.title}입니다."
        )
    else:
        intro_bullets = [
            f"**Date**: {daily.date}",
            "**Source**: Hugging Face Daily Papers",
            f"**Papers**: {len(daily.papers)} Papers Included",
        ]
        intro_body = _text_slide_body("Daily AI Papers", intro_bullets)
        intro_script = (
            f"Hi Everyone. This is NLP Corgi. "
            f"Daily AI Papers for {daily.date}. "
            f"This episode covers {len(daily.papers)} papers from Hugging Face Daily Papers."
        )

    _add_slide(
        lines,
        scripts,
        intro_body,
        intro_script,
    )

    for idx, paper in enumerate(daily.papers, 1):
        paper_library = figure_libraries.get(paper.paper_id) if figure_libraries else None

        for slide_idx, slide in enumerate(paper.slides, 1):
            asset = _resolve_figure_asset(slide, paper_library)
            _attach_figure_to_slide(slide, asset, out_dir, idx, slide_idx)
            _add_slide(
                lines,
                scripts,
                _llm_slide_body(idx, slide_idx, slide, single_pdf_mode),
                slide.script or " ".join(slide.bullets),
            )

    if lines and lines[-1] == "---":
        lines.pop()

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    return scripts
