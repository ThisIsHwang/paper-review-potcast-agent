import argparse
import datetime
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


def _ensure_packages_distributions() -> None:
    """Monkey-patch importlib.metadata.packages_distributions for Python <3.10."""
    import importlib.metadata as _im

    if hasattr(_im, "packages_distributions"):
        return
    try:
        import importlib_metadata as _backport

        _im.packages_distributions = _backport.packages_distributions  # type: ignore[attr-defined, assignment]
    except Exception as exc:  # pragma: no cover - best-effort fallback
        import warnings

        warnings.warn(
            f"packages_distributions missing and backport unavailable: {exc}. "
            "Upgrade to Python 3.10+ for full compatibility.",
            RuntimeWarning,
        )


_ensure_packages_distributions()

from config import Config, load_config
from daily_papers.hf_client import fetch_daily_papers, resolve_arxiv_and_pdf
from daily_papers.models import HFPaperEntry
from daily_papers.pdf_downloader import download_pdf
from daily_papers.pdf_parser import extract_core_text
from llm.client import LLMClient
from llm.summarizer import DailyEpisode, PaperSummary, summarize_paper
from llm.translator import language_display, translate_scripts
from slides.figure_assets import FigureLibrary, load_figure_library, summarize_assets
from slides.figure_extractor import extract_pdf_figures
from slides.markdown_builder import build_daily_markdown
from slides.marp_renderer import render_markdown_to_images
from storage import paths
from tts.client import TTSClient
from video.builder import build_video
from youtube.uploader import upload_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily AI Papers → YouTube automation")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--top-k", type=int, dest="top_k", help="How many papers to include.")
    parser.add_argument(
        "--languages",
        help="Comma-separated narration languages (e.g., en,ko). Defaults to LANGUAGES env/config.",
    )
    parser.add_argument("--pdf-url", help="Process a single PDF by URL (bypasses Daily Papers fetch).")
    parser.add_argument("--paper-id", help="Optional paper id when using --pdf-url.")
    parser.add_argument("--paper-title", help="Optional paper title when using --pdf-url.")
    parser.add_argument("--origin", help="Optional origin/affiliation override for single PDF mode.")
    parser.add_argument("--skip-render", action="store_true", help="Skip marp slide rendering.")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS generation.")
    parser.add_argument("--skip-video", action="store_true", help="Skip video rendering.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip YouTube upload.")
    parser.add_argument("--video-only", action="store_true", help="Produce video locally and skip YouTube upload.")
    return parser.parse_args()


def build_description(papers: List[Any], base_url: str) -> str:
    lines = ["Daily AI Papers", "", "Source: Hugging Face Daily Papers", ""]
    for idx, paper in enumerate(papers, 1):
        title = getattr(paper, "title", "Untitled")
        authors = ", ".join(getattr(paper, "authors", []))
        link = getattr(paper, "arxiv_url", "") or getattr(paper, "hf_url", "") or base_url
        lines.append(f"{idx}. {title}")
        lines.append(f"   - Authors: {authors}")
        lines.append(f"   - Link: {link}")
        lines.append("")
    return "\n".join(lines)


def _normalize_languages(raw_list: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lang in raw_list:
        code = lang.strip().lower()
        if code and code not in seen:
            result.append(code)
            seen.add(code)
    return result


def _parse_languages(cli_value: Optional[str], config_langs: List[str]) -> List[str]:
    if cli_value:
        langs = _normalize_languages(cli_value.split(","))
        if langs:
            return langs
    langs = _normalize_languages(config_langs)
    return langs or ["en"]


def _strip_leading_enumeration(text: str) -> str:
    return re.sub(r"^\s*\d+\s*[.\)-:]?\s*", "", text.strip())


def _strip_delivery_cues(text: str) -> str:
    cleaned = re.sub(r"\[[^\[\]]+?\]", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _normalize_texts(items: List[str]) -> List[str]:
    return [_strip_delivery_cues(_strip_leading_enumeration(s)) for s in items]


def _clean_text(text: str) -> str:
    return text.strip().strip(' "“”') if text else ""


def _paper_id_from_pdf_url(url: str) -> str:
    path = url.split("?")[0].split("#")[0]
    last = path.rstrip("/").split("/")[-1] if path else "custom_pdf"
    last = last.replace(".pdf", "") if last.endswith(".pdf") else last
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", last).strip("_") or "custom_pdf"
    return cleaned


def _write_scripts_file(scripts: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, script in enumerate(scripts, 1):
            clean_script = _strip_delivery_cues(_strip_leading_enumeration(script))
            f.write(f"[Slide {idx}]\n")
            f.write("Script:\n")
            f.write(clean_script.strip())
            f.write("\n\n")


def _setup_logging(config: Config) -> None:
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _load_input_papers(
    args: argparse.Namespace,
    target_date: str,
    top_k: int,
    config: Config,
    origin_override: Optional[str],
) -> Tuple[List[HFPaperEntry], bool]:
    if args.pdf_url:
        paper_id = args.paper_id or _paper_id_from_pdf_url(args.pdf_url)
        title = args.paper_title or f"Custom PDF ({paper_id})"
        paper = HFPaperEntry(
            paper_id=paper_id,
            title=title,
            summary=f"Custom PDF input from {args.pdf_url}",
            authors=[],
            upvotes=0,
            published_at=target_date,
            pdf_url=args.pdf_url,
            arxiv_url=None,
            hf_url=None,
            origin=origin_override,
        )
        logging.info("Single PDF mode: processing %s (%s)", paper_id, args.pdf_url)
        return [paper], True

    papers = fetch_daily_papers(target_date, top_k, config.hf_base_url)
    return papers, False


def _summarize_papers(
    papers: List[HFPaperEntry],
    target_date: str,
    config: Config,
    llm_client: LLMClient,
    single_pdf_mode: bool,
    origin_override: Optional[str],
) -> Tuple[List[PaperSummary], Dict[str, FigureLibrary]]:
    figure_libraries: Dict[str, FigureLibrary] = {}
    figure_summaries: Dict[str, List[str]] = {}
    summaries: List[PaperSummary] = []

    for paper in papers:
        try:
            resolved_paper = resolve_arxiv_and_pdf(paper)
            pdf_path = download_pdf(resolved_paper, target_date, config.output_base_dir)
            if not pdf_path:
                continue

            captions_path = extract_pdf_figures(pdf_path)
            figure_lib = load_figure_library(captions_path) if captions_path else None
            if figure_lib:
                figure_libraries[resolved_paper.paper_id] = figure_lib
                figure_summaries[resolved_paper.paper_id] = summarize_assets(figure_lib, limit=20)

            extracted = extract_core_text(pdf_path)
            summary = summarize_paper(
                resolved_paper,
                extracted,
                llm_client,
                figure_summaries=figure_summaries.get(resolved_paper.paper_id),
            )
            if single_pdf_mode and origin_override:
                summary.origin = origin_override
            summaries.append(summary)

        except Exception as exc:
            logging.exception("Failed to process paper %s: %s", paper.paper_id, exc)

    return summaries, figure_libraries


def _build_scripts_by_language(
    scripts: List[str],
    target_languages: List[str],
    llm_client: LLMClient,
    single_pdf_mode: bool,
    daily: DailyEpisode,
) -> Dict[str, List[str]]:
    scripts_by_lang: Dict[str, List[str]] = {"en": scripts}

    for lang in target_languages:
        if lang == "en":
            continue
        try:
            translated_scripts = translate_scripts(scripts, llm_client, lang)
            scripts_by_lang[lang] = _normalize_texts(translated_scripts)
            logging.info("Translated scripts to %s", language_display(lang))
        except Exception as exc:
            logging.exception("Failed to translate scripts to %s: %s", lang, exc)

    if single_pdf_mode and daily.papers:
        paper = daily.papers[0]
        origin = _clean_text(getattr(paper, "origin", ""))
        for lang in target_languages:
            if lang == "ko":
                origin_phrase = f"{origin}에서 나온 " if origin else ""
                intro_script = (
                    f"안녕하세요. NLP 코기입니다. 오늘의 꼬리의 꼬리를 무는 페이퍼 딥다이브, "
                    f"꼬꼬페에서 다룰 논문은 {origin_phrase}{paper.title}입니다."
                )
            else:
                origin_phrase = f"from {origin} " if origin else ""
                intro_script = (
                    f"Hi everyone. This is NLP Corgi. Today's deep dive is about the paper {origin_phrase}"
                    f"{paper.title}."
                )
            if scripts_by_lang.get(lang):
                scripts_by_lang[lang][0] = intro_script

    return scripts_by_lang


def _write_scripts_by_language(
    scripts_by_lang: Dict[str, List[str]],
    target_date: str,
    output_base_dir: str,
    single_paper_id: Optional[str],
) -> None:
    for lang, lang_scripts in scripts_by_lang.items():
        scripts_path = paths.scripts_path(output_base_dir, target_date, lang, paper_id=single_paper_id)
        _write_scripts_file(lang_scripts, scripts_path)
        logging.info("Saved %s scripts to %s", language_display(lang), scripts_path)


def _render_images(
    args: argparse.Namespace,
    md_path: str,
    output_base_dir: str,
    target_date: str,
    single_paper_id: Optional[str],
) -> List[str]:
    if args.skip_render:
        logging.info("Skipping marp rendering step.")
        return []

    image_paths = render_markdown_to_images(
        md_path,
        paths.slide_prefix(output_base_dir, target_date, paper_id=single_paper_id),
    )
    if not image_paths:
        logging.error("No slide images generated.")
    return image_paths


def _generate_audio(
    args: argparse.Namespace,
    scripts_by_lang: Dict[str, List[str]],
    target_languages: List[str],
    config: Config,
    target_date: str,
    single_paper_id: Optional[str],
) -> Dict[str, List[str]]:
    if args.skip_tts:
        logging.info("Skipping TTS generation.")
        return {}

    tts_client = TTSClient(
        api_key=config.openai_api_key,
        model=config.openai_tts_model,
        voice=config.openai_tts_voice,
        style_instruction=config.tts_style_instruction,
        speed=config.tts_speed,
    )

    audio_by_lang: Dict[str, List[str]] = {}
    for lang in target_languages:
        scripts_for_lang = scripts_by_lang.get(lang)
        if not scripts_for_lang:
            logging.warning("No scripts available for language %s; skipping TTS.", lang)
            continue

        audio_dir = paths.audio_lang_dir(config.output_base_dir, target_date, lang, paper_id=single_paper_id)
        audio_files = tts_client.synthesize_scripts(scripts_for_lang, audio_dir)
        if audio_files:
            audio_by_lang[lang] = audio_files
        else:
            logging.error("No audio files generated for language %s.", lang)

    return audio_by_lang


def _build_videos(
    args: argparse.Namespace,
    target_languages: List[str],
    primary_language: str,
    image_paths: List[str],
    audio_by_lang: Dict[str, List[str]],
    scripts_by_lang: Dict[str, List[str]],
    output_base_dir: str,
    target_date: str,
    single_paper_id: Optional[str],
) -> Dict[str, str]:
    if args.skip_video:
        logging.info("Skipping video rendering.")
        return {}

    video_paths: Dict[str, str] = {}
    for lang in target_languages:
        audio_files = audio_by_lang.get(lang)
        if image_paths and audio_files and len(image_paths) == len(audio_files):
            lang_suffix = None if lang == primary_language else lang
            video_path = paths.video_lang_path(output_base_dir, target_date, lang_suffix, paper_id=single_paper_id)

            subtitle_scripts = scripts_by_lang.get(lang)
            if subtitle_scripts is not None and len(subtitle_scripts) != len(audio_files):
                logging.warning(
                    "Skipping subtitles for %s due to script/audio mismatch (scripts=%d, audio=%d).",
                    language_display(lang),
                    len(subtitle_scripts),
                    len(audio_files),
                )
                subtitle_scripts = None

            build_video(image_paths, audio_files, video_path, subtitle_scripts=subtitle_scripts)
            video_paths[lang] = video_path
            continue

        logging.error(
            "Cannot build video for %s: missing images or audio, or count mismatch.",
            language_display(lang),
        )

    return video_paths


def _upload_videos(
    video_paths: Dict[str, str],
    skip_upload: bool,
    config: Config,
    papers: List[HFPaperEntry],
    target_date: str,
    summaries_count: int,
    primary_language: str,
) -> None:
    if skip_upload:
        logging.info("Upload skipped by flag.")
        return

    if not video_paths:
        logging.error("No videos built to upload.")
        return

    if not (config.youtube_client_secrets and config.youtube_token_file):
        logging.warning("YouTube credentials missing. Skipping upload.")
        return

    description = build_description(papers, config.hf_base_url)
    multi_language = len(video_paths) > 1

    for lang, video_path in video_paths.items():
        title_suffix = "" if (lang == primary_language and not multi_language) else f" [{language_display(lang)}]"
        title = f"Daily AI Papers - {target_date} | Top {summaries_count} ML Papers{title_suffix}"
        tags = [target_date, "AI", "Machine Learning", "Research", "Daily Papers", language_display(lang)]
        per_lang_desc = description if not multi_language else f"Language: {language_display(lang)}\n\n{description}"
        upload_video(
            video_path,
            title,
            per_lang_desc,
            tags,
            config.youtube_client_secrets,
            config.youtube_token_file,
        )


def main() -> int:
    args = parse_args()
    config = load_config()
    _setup_logging(config)

    target_date = args.date or datetime.date.today().isoformat()
    top_k = args.top_k or config.top_k
    target_languages = _parse_languages(args.languages, config.languages)
    primary_language = target_languages[0] if target_languages else "en"
    origin_override = _clean_text(args.origin) if args.origin else None

    logging.info("Starting pipeline for %s (top_k=%d)", target_date, top_k)
    logging.info("Narration languages: %s", ", ".join(target_languages))
    if not config.openai_api_key:
        logging.warning("OPENAI_API_KEY not set. LLM/TTS calls may fail.")

    papers, single_pdf_mode = _load_input_papers(args, target_date, top_k, config, origin_override)
    if not papers:
        logging.error("No papers retrieved. Exiting.")
        return 1

    llm_client = LLMClient(api_key=config.openai_api_key, model=config.openai_llm_model)
    summaries, figure_libraries = _summarize_papers(
        papers,
        target_date,
        config,
        llm_client,
        single_pdf_mode,
        origin_override,
    )
    if not summaries:
        logging.error("No summaries created. Exiting.")
        return 1

    daily = DailyEpisode(date=target_date, papers=summaries)
    single_paper_id = daily.papers[0].paper_id if single_pdf_mode and daily.papers else None

    md_path = paths.markdown_path(config.output_base_dir, target_date, paper_id=single_paper_id)
    scripts = build_daily_markdown(
        daily,
        md_path,
        figure_libraries,
        single_pdf_mode=single_pdf_mode,
    )
    scripts = _normalize_texts(scripts)
    logging.info("Markdown created at %s", md_path)

    scripts_by_lang = _build_scripts_by_language(
        scripts,
        target_languages,
        llm_client,
        single_pdf_mode,
        daily,
    )
    _write_scripts_by_language(scripts_by_lang, target_date, config.output_base_dir, single_paper_id)

    image_paths = _render_images(args, md_path, config.output_base_dir, target_date, single_paper_id)
    audio_by_lang = _generate_audio(args, scripts_by_lang, target_languages, config, target_date, single_paper_id)
    video_paths = _build_videos(
        args,
        target_languages,
        primary_language,
        image_paths,
        audio_by_lang,
        scripts_by_lang,
        config.output_base_dir,
        target_date,
        single_paper_id,
    )

    _upload_videos(
        video_paths,
        skip_upload=(args.skip_upload or args.video_only),
        config=config,
        papers=papers,
        target_date=target_date,
        summaries_count=len(summaries),
        primary_language=primary_language,
    )

    logging.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
