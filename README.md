# AutoYouTube

Automated pipeline that turns **Hugging Face Daily Papers** (or a single PDF) into:

- slide deck images,
- narrated audio tracks,
- subtitle-burned MP4 videos,
- optional YouTube uploads.

The repository is built for daily research-content production with minimal manual steps.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [CLI Reference](#cli-reference)
- [Environment Variables](#environment-variables)
- [Outputs](#outputs)
- [YouTube Upload Behavior](#youtube-upload-behavior)
- [Figure Extraction (Optional but Built-in)](#figure-extraction-optional-but-built-in)
- [Troubleshooting](#troubleshooting)
- [Publishing Checklist](#publishing-checklist)
- [Known Limitations](#known-limitations)
- [License](#license)

## Overview

`main.py` orchestrates a full content pipeline:

1. Fetch top papers from Hugging Face Daily Papers (or use `--pdf-url`).
2. Resolve PDF links and download PDFs.
3. Extract core paper text (`abstract`, `introduction`, `conclusion`, plus full text).
4. Ask an LLM for structured slide content in JSON.
5. Build Marp markdown with modern CSS and optional figure embedding.
6. Render slides to PNG.
7. Generate TTS audio per slide and per language.
8. Compose video with subtitles using MoviePy.
9. Upload to YouTube (optional).

## Pipeline

```text
Hugging Face Daily Papers / --pdf-url
  -> PDF download
  -> PDF text + figure extraction
  -> LLM summarization -> slide specs
  -> Marp markdown generation
  -> Marp PNG rendering
  -> OpenAI TTS (per slide, per language)
  -> MoviePy composition + hard subtitles
  -> MP4 output
  -> Optional YouTube upload
```

## Tech Stack

- Python (pipeline orchestration)
- OpenAI API (LLM summarization, translation, TTS)
- PyMuPDF (PDF text extraction)
- Marp CLI (slide rendering)
- MoviePy + FFmpeg (video composition/encoding)
- YouTube Data API v3 (upload)
- Optional: LayoutParser + Detectron2 (figure/table extraction)

## Repository Structure

```text
main.py                  # Entry point
config.py                # Env-driven configuration

daily_papers/            # HF fetch + PDF handling + text extraction
llm/                     # LLM client, prompts, summarizer, translator
slides/                  # Figure handling, markdown builder, Marp renderer
tts/                     # OpenAI TTS client
video/                   # MoviePy video builder + subtitles
youtube/                 # YouTube OAuth/upload
storage/                 # Output path helpers
scripts/                 # Setup and convenience scripts
models/publaynet/        # PubLayNet config + weights (for figure extraction)
```

## Requirements

### Runtime

- Python `3.10+` recommended
- `ffmpeg` on `PATH`
- `marp` CLI on `PATH`
- OpenAI API key

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Install system dependencies

```bash
# macOS example
brew install ffmpeg node
npm i -g @marp-team/marp-cli
```

## Quick Start

### 1. Create virtual environment

```bash
./scripts/setup_venv.sh
```

### 2. Configure environment

```bash
cp .env.example .env
```

Set at least:

- `OPENAI_API_KEY`

If uploading to YouTube, also set:

- `YOUTUBE_CLIENT_SECRETS_FILE`
- `YOUTUBE_TOKEN_FILE`

Optional but useful to add in `.env`:

```bash
LANGUAGES=en,ko
TTS_SPEED=1.2
TTS_STYLE_INSTRUCTION=
```

### 3. Activate environment

```bash
source scripts/env.sh
```

### 4. Run pipeline (local video build, no upload)

```bash
python main.py --date 2026-02-21 --top-k 10 --skip-upload
```

## Usage

### Daily papers mode

```bash
python main.py --date 2026-02-21 --top-k 10 --skip-upload
```

### Video-only shortcut (always skips upload)

```bash
python main.py --date 2026-02-21 --top-k 10 --video-only
# or
./scripts/run_video_only.sh 2026-02-21 10
```

### Single PDF mode

```bash
python main.py \
  --pdf-url https://arxiv.org/pdf/2511.21689.pdf \
  --paper-id 2511.21689 \
  --paper-title "Your Paper Title" \
  --origin "Your Lab/Company" \
  --skip-upload
```

### Partial runs (debug/dev)

```bash
python main.py --date 2026-02-21 --skip-render
python main.py --date 2026-02-21 --skip-tts
python main.py --date 2026-02-21 --skip-video
```

## CLI Reference

| Flag | Description |
| --- | --- |
| `--date YYYY-MM-DD` | Target date. Defaults to today. |
| `--top-k N` | Number of papers to process. Defaults to `TOP_K`/`10`. |
| `--languages en,ko,...` | Narration languages. First language is primary output video. |
| `--pdf-url URL` | Process a single PDF instead of Daily Papers fetch. |
| `--paper-id ID` | Optional custom ID in single-PDF mode. |
| `--paper-title TITLE` | Optional custom title in single-PDF mode. |
| `--origin TEXT` | Optional affiliation/origin override for intro narration. |
| `--skip-render` | Skip Marp PNG rendering. |
| `--skip-tts` | Skip TTS generation. |
| `--skip-video` | Skip video composition. |
| `--skip-upload` | Disable YouTube upload. |
| `--video-only` | Build local video and skip upload (same effect as `--skip-upload` for upload phase). |

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `HF_BASE_URL` | `https://huggingface.co` | Hugging Face base URL. |
| `TOP_K` | `10` | Default top-K paper count. |
| `OPENAI_API_KEY` | - | Required for LLM/TTS steps. |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | Model used for summarization and translation. |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts` | TTS model. |
| `OPENAI_TTS_VOICE` | `alloy` | Voice preset. |
| `TTS_STYLE_INSTRUCTION` | empty | Style hint string (stored, currently not injected into spoken text). |
| `TTS_SPEED` | `1.2` | TTS speed multiplier, clamped to `[0.5, 4.0]`. |
| `LANGUAGES` | `en,ko` | Comma-separated target narration languages. |
| `YOUTUBE_CLIENT_SECRETS_FILE` | empty | OAuth client JSON path. |
| `YOUTUBE_TOKEN_FILE` | empty | OAuth token cache path. |
| `OUTPUT_BASE_DIR` | `./outputs` | Base output directory. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

## Outputs

### Daily papers mode (`--date`)

```text
outputs/{date}/
  daily_papers_{date}.mp4
  daily_papers_{date}_{lang}.mp4
  slides/
    slides_{date}.md
    slides_{date}_*.png
    scripts_{date}_{lang}.txt
  {paper_id}/
    paper.pdf
    captions.json
    figures/*.png
  audio/{lang}/
    audio_slide_001.mp3
    ...
```

### Single PDF mode (`--pdf-url`)

```text
outputs/{date}/{paper_id}/
  paper.pdf
  captions.json
  figures/*.png
  slides/
    slides_{date}.md
    slides_{date}_*.png
    figures/*.png          # copied assets used in slide markdown
  scripts_{date}_{lang}.txt
  audio/{lang}/audio_slide_*.mp3
  daily_papers_{date}.mp4
  daily_papers_{date}_{lang}.mp4
```

## YouTube Upload Behavior

- Upload occurs only when both conditions are met:
  - you did not pass `--skip-upload` / `--video-only`
  - both `YOUTUBE_CLIENT_SECRETS_FILE` and `YOUTUBE_TOKEN_FILE` are configured
- First OAuth run opens a local browser for consent.
- Videos are uploaded as:
  - `privacyStatus: unlisted`
  - `categoryId: 28` (Science & Technology)
- In multi-language output, title suffix includes language labels (for non-primary tracks).

## Figure Extraction (Optional but Built-in)

Figure extraction runs per downloaded PDF before summarization.

Behavior:

- If `captions.json` already exists, extraction is skipped (cache behavior).
- If model files or optional deps are missing, extraction is skipped gracefully.
- Extracted figure/table metadata is fed into the LLM prompt to improve slide quality.

Model/env paths:

- Default model files:
  - `models/publaynet/config.yaml`
  - `models/publaynet/model_final.pth`
- Override via:
  - `PUBLayNET_CONFIG`
  - `PUBLayNET_WEIGHTS`

Optional dependencies for this feature are not in `requirements.txt` by default (e.g., `layoutparser`, `detectron2`, `opencv-python`, `numpy`).

## Troubleshooting

### `marp CLI not found`

Install Marp CLI globally and verify:

```bash
npm i -g @marp-team/marp-cli
marp --version
```

### FFmpeg / video encoding errors

Install and verify:

```bash
ffmpeg -version
```

### OpenAI request failures

Check:

- `OPENAI_API_KEY` is set correctly
- model names are valid for your account
- quota/rate limits are not exhausted

### No video built

Common causes:

- slide image count and audio file count mismatch
- `--skip-render` or `--skip-tts` used in ways that leave missing artifacts
- upstream LLM/TTS failures in logs

### YouTube upload errors

Check:

- OAuth client JSON path is valid
- token path is writable
- if token is stale, delete token JSON and authenticate again

## Publishing Checklist

Before pushing to GitHub:

- Remove or ignore secrets:
  - `.env`
  - OAuth token files
  - OAuth client secrets
- Avoid committing large generated artifacts:
  - `outputs/`
  - local MP4/PNG/MP3 files
- Add a proper `.gitignore` if not present.
- Keep `LICENSE` and `NOTICE` files when redistributing.

## Known Limitations

- No automated test suite is included yet.
- LLM output quality depends heavily on model behavior and prompt adherence.
- TTS is generated slide-by-slide; long runs can be costly/time-consuming.
- Figure extraction quality depends on external model/deps and PDF layout quality.

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.

If you redistribute this project or derivative works, you should:

- include the `LICENSE` file,
- retain the `NOTICE` file,
- keep attribution to the original project/author in source or documentation.
