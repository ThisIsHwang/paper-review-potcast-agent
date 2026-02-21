import os
from typing import Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def date_dir(base_dir: str, date: str) -> str:
    path = os.path.join(base_dir, date)
    ensure_dir(path)
    return path


def paper_dir(base_dir: str, date: str, paper_id: str) -> str:
    path = os.path.join(date_dir(base_dir, date), paper_id)
    ensure_dir(path)
    return path


def slides_dir(base_dir: str, date: str, paper_id: Optional[str] = None) -> str:
    base = paper_dir(base_dir, date, paper_id) if paper_id else date_dir(base_dir, date)
    path = os.path.join(base, "slides")
    ensure_dir(path)
    return path


def audio_dir(base_dir: str, date: str, paper_id: Optional[str] = None) -> str:
    base = paper_dir(base_dir, date, paper_id) if paper_id else date_dir(base_dir, date)
    path = os.path.join(base, "audio")
    ensure_dir(path)
    return path


def audio_lang_dir(base_dir: str, date: str, lang: str, paper_id: Optional[str] = None) -> str:
    path = os.path.join(audio_dir(base_dir, date, paper_id), lang)
    ensure_dir(path)
    return path


def video_path(base_dir: str, date: str, paper_id: Optional[str] = None) -> str:
    base = paper_dir(base_dir, date, paper_id) if paper_id else date_dir(base_dir, date)
    return os.path.join(base, f"daily_papers_{date}.mp4")


def markdown_path(base_dir: str, date: str, paper_id: Optional[str] = None) -> str:
    return os.path.join(slides_dir(base_dir, date, paper_id), f"slides_{date}.md")


def slide_prefix(base_dir: str, date: str, paper_id: Optional[str] = None) -> str:
    return os.path.join(slides_dir(base_dir, date, paper_id), f"slides_{date}_")


def scripts_path(base_dir: str, date: str, lang: str, paper_id: Optional[str] = None) -> str:
    base = paper_dir(base_dir, date, paper_id) if paper_id else slides_dir(base_dir, date, paper_id)
    return os.path.join(base, f"scripts_{date}_{lang}.txt")


def video_lang_path(base_dir: str, date: str, lang: Optional[str], paper_id: Optional[str] = None) -> str:
    base = paper_dir(base_dir, date, paper_id) if paper_id else date_dir(base_dir, date)
    suffix = f"_{lang}" if lang else ""
    return os.path.join(base, f"daily_papers_{date}{suffix}.mp4")
