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
import os
import re
import textwrap
from typing import List, Optional, Sequence, Tuple

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ColorClip, ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_SUBTITLE_CHAR_MAP = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u2043": "-",  # hyphen bullet
        "\uFE58": "-",  # small em dash
        "\uFE63": "-",  # small hyphen-minus
        "\uFF0D": "-",  # fullwidth hyphen-minus
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u00A0": " ",  # no-break space
        "\u202F": " ",  # narrow no-break space
        "\u200B": "",  # zero-width space
        "\u200C": "",  # zero-width non-joiner
        "\u200D": "",  # zero-width joiner
        "\uFEFF": "",  # BOM
    }
)
_FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _find_subtitle_font() -> Optional[str]:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


_SUBTITLE_FONT = _find_subtitle_font()


def _clean_subtitle_text(text: str) -> str:
    cleaned = (text or "").translate(_SUBTITLE_CHAR_MAP)
    cleaned = re.sub(r"\[[^\[\]]+?\]", "", cleaned)
    cleaned = re.sub(r"^\s*\d+\s*[.\)-:]?\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _split_subtitle_units(text: str, max_chars: int = 78) -> List[str]:
    normalized = _clean_subtitle_text(text)
    if not normalized:
        return []

    units: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        wrapped = textwrap.wrap(
            sentence,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
        units.extend([chunk.strip() for chunk in wrapped if chunk.strip()])

    return units or [normalized]


def _time_segments(text: str, duration: float) -> List[Tuple[float, float, str]]:
    units = _split_subtitle_units(text)
    if not units or duration <= 0:
        return []

    weights = [max(len(re.sub(r"\s+", "", unit)), 1) for unit in units]
    total_weight = max(sum(weights), 1)
    raw_durations = [duration * (weight / total_weight) for weight in weights]

    min_duration = min(0.9, duration / max(len(units), 1))
    adjusted = [max(min_duration, seg_duration) for seg_duration in raw_durations]
    scale = duration / sum(adjusted)
    adjusted = [seg_duration * scale for seg_duration in adjusted]

    segments: List[Tuple[float, float, str]] = []
    start = 0.0
    for idx, (unit, seg_duration) in enumerate(zip(units, adjusted)):
        if idx == len(units) - 1:
            seg_duration = max(duration - start, 0.0)
        segments.append((start, seg_duration, unit))
        start += seg_duration
    return segments


def _make_subtitle_text_clip(
    text: str,
    font_size: int,
    font: Optional[str],
    max_text_width: int,
) -> TextClip:
    style_kwargs = {
        "font": font,
        "font_size": font_size,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 2,
        "text_align": "center",
    }
    clip = TextClip(text=text, method="label", **style_kwargs)
    if clip.w <= max_text_width:
        return clip

    clip.close()
    estimated_chars = max(12, int(len(text) * max_text_width / max(1, clip.w)))
    wrapped_text = text
    for _ in range(6):
        wrapped_text = "\n".join(
            textwrap.wrap(
                text,
                width=estimated_chars,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
        wrapped = TextClip(text=wrapped_text, method="label", **style_kwargs)
        if wrapped.w <= max_text_width or estimated_chars <= 8:
            return wrapped
        ratio = max_text_width / max(1, wrapped.w)
        estimated_chars = max(8, int(estimated_chars * ratio) - 1)
        wrapped.close()

    return TextClip(text=wrapped_text, method="label", **style_kwargs)


def _build_subtitle_text_clip(
    text: str,
    font_size: int,
    font: Optional[str],
    max_text_width: int,
) -> TextClip:
    try:
        return _make_subtitle_text_clip(text, font_size, font, max_text_width)
    except Exception as exc:
        if font:
            logging.warning("Failed to use subtitle font %s (%s). Falling back to default font.", font, exc)
            return _make_subtitle_text_clip(text, font_size, None, max_text_width)
        raise


def _positioned_subtitle_layers(
    text_clip: TextClip,
    start: float,
    duration: float,
    video_size: Tuple[int, int],
) -> Tuple[ColorClip, TextClip]:
    width, height = video_size
    pad_x = max(int(height * 0.014), 16)
    pad_y = max(int(height * 0.01), 10)
    bottom_margin = max(int(height * 0.04), 28)

    bg_width = min(text_clip.w + (pad_x * 2), int(width * 0.94))
    bg_height = text_clip.h + (pad_y * 2)
    y_pos = height - bottom_margin - bg_height

    bg_clip = (
        ColorClip(size=(bg_width, bg_height), color=(0, 0, 0))
        .with_opacity(0.55)
        .with_start(start)
        .with_duration(duration)
        .with_position(("center", y_pos))
    )

    timed_text = (
        text_clip.with_start(start)
        .with_duration(duration)
        .with_position(("center", y_pos + pad_y))
    )
    return bg_clip, timed_text


def _subtitle_layers(text: str, duration: float, video_size: Tuple[int, int]) -> List[object]:
    width, height = video_size
    font_size = max(int(height * 0.04), 24)
    max_text_width = max(int(width * 0.86), 220)

    layers: List[object] = []
    for start, seg_duration, segment_text in _time_segments(text, duration):
        if seg_duration <= 0:
            continue

        try:
            text_clip = _build_subtitle_text_clip(
                segment_text,
                font_size,
                _SUBTITLE_FONT,
                max_text_width,
            )
            bg_clip, timed_text = _positioned_subtitle_layers(text_clip, start, seg_duration, video_size)
        except Exception as exc:
            logging.warning("Subtitle text rendering failed: %s", exc)
            continue

        layers.extend([bg_clip, timed_text])

    return layers


def build_video(
    slide_images: List[str],
    audio_files: List[str],
    out_path: str,
    subtitle_scripts: Optional[Sequence[str]] = None,
) -> None:
    if len(slide_images) != len(audio_files):
        raise ValueError("slide_images and audio_files length mismatch")
    if subtitle_scripts is not None and len(subtitle_scripts) != len(audio_files):
        raise ValueError("subtitle_scripts and audio_files length mismatch")

    subtitle_enabled = subtitle_scripts is not None
    logging.info("Building video with %d slides%s", len(slide_images), " + subtitles" if subtitle_enabled else "")

    clips = []
    for idx, (img, audio) in enumerate(zip(slide_images, audio_files)):
        a_clip = AudioFileClip(audio)
        base_clip = ImageClip(img).with_duration(a_clip.duration).with_audio(a_clip)

        clip = base_clip
        if subtitle_scripts is not None:
            try:
                layers = _subtitle_layers(subtitle_scripts[idx], a_clip.duration, base_clip.size)
                if layers:
                    clip = CompositeVideoClip([base_clip, *layers], size=base_clip.size).with_duration(a_clip.duration)
                    clip = clip.with_audio(a_clip)
            except Exception as exc:
                logging.warning("Subtitle rendering failed for slide %d: %s", idx + 1, exc)

        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    try:
        final.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac")
    finally:
        final.close()
        for clip in clips:
            clip.close()
    logging.info("Video saved to %s", out_path)
