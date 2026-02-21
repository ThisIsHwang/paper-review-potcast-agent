# 📄 Daily AI Papers → YouTube 자동 업로드 시스템 명세서 (v0.1)

## 0. 목표

* Hugging Face **Daily Papers**에서 **매일 upvote 기준 Top N(기본 10)** 논문을 가져온다.
* 각 논문에 대해:

  * 핵심 아이디어/인사이트를 LLM으로 요약하고
  * 슬라이드용 자료(텍스트 구조 + 이미지)를 자동 생성하고
  * 슬라이드별 나레이션(TTS 오디오)을 생성한 뒤
  * **슬라이드 이미지 + 오디오를 합쳐 하나의 영상(mp4)**을 만든다.
* 최종적으로 **YouTube 채널에 매일 자동 업로드**한다.
* MVP 버전에서는 **하루 1개 영상(Top N 논문 합본 영상)**을 생성한다.

---

## 1. 기술 스택 & 환경

* 언어: Python 3.10+
* 필수 라이브러리(예시)

  * HTTP: `requests`
  * PDF 처리: `pymupdf` (`fitz`)
  * LLM: OpenAI Python SDK (또는 추상화 인터페이스)
  * TTS: OpenAI TTS API (또는 다른 TTS 클라이언트, 인터페이스로 추상화)
  * 슬라이드:

    * Markdown 템플릿 생성 (직접 구현)
    * 외부 CLI: `@marp-team/marp-cli` (Node 기반, 시스템에 설치되어 있다고 가정)
  * 비디오: `moviepy`
  * YouTube 업로드: `google-api-python-client`, `google-auth-oauthlib`
* 실행:

  * CLI 명령: `python main.py --date YYYY-MM-DD`
  * 스케줄링은 cron / GitHub Actions / Cloud Scheduler 등 외부에서 호출

---

## 2. 전체 플로우

1. **입력 날짜 결정**

   * `--date` 인자가 없으면 오늘 날짜(로컬 기준) 사용.

2. **HF Daily Papers 조회**

   * `/api/daily_papers?date={date}` 호출.
   * 응답 리스트에서 `upvotes` 기준 Top N 추출.

3. **각 논문에 대한 메타데이터 & PDF 확보**

   * paper id (arxiv id)를 이용해 PDF URL 구성 또는 Paper 페이지에서 추출.
   * PDF 다운로드 후 로컬 캐시 폴더에 저장.

4. **PDF에서 핵심 텍스트 추출**

   * `Abstract`, `Introduction`, `Conclusion` 위주로 텍스트 뽑기.
   * (선택) Figure가 많은 페이지를 탐지하여 이미지 추출.

5. **LLM으로 논문 요약 → 구조화된 JSON 생성**

   * 논문 하나당:

     * 한 줄 요약
     * 핵심 key ideas
     * 인사이트 (Why it matters, Limitations 등)
     * 슬라이드 구조 (슬라이드 제목, bullet, 슬라이드별 script)
   * 이 JSON을 내부 공통 스키마로 사용.

6. **슬라이드(발표자료) 마크다운 생성**

   * 여러 논문의 슬라이드를 **하나의 슬라이드 데크**(Marp 마크다운)로 생성.
   * `slides_{date}.md` 파일 생성.

7. **Marp로 슬라이드 이미지 생성**

   * `marp slides_{date}.md --images slides_{date}_` 호출.
   * `slides_{date}_001.png`, `slides_{date}_002.png` … 형태로 생성.

8. **슬라이드별 TTS 오디오 생성**

   * LLM이 준 `per_slide script` 기준으로

     * `audio_slide_001.mp3`, `audio_slide_002.mp3` … 생성.

9. **슬라이드 이미지 + 오디오 → mp4 렌더링**

   * 같은 인덱스의 슬라이드 이미지와 오디오를 매칭.
   * 영상 타임라인 상에서

     * `slide_i.png`는 `audio_i`의 duration만큼 보여준다.
   * 모든 슬라이드를 이어 붙여 `daily_papers_{date}.mp4` 생성.

10. **YouTube 자동 업로드**

    * 제목, 설명, 태그를 LLM 또는 템플릿으로 생성.
    * YouTube Data API `videos.insert`로 업로드.
    * 업로드 성공 시 video id 로그 및 파일/DB에 기록.

---

## 3. 디렉터리 구조 제안

```text
project_root/
  main.py
  config.py
  requirements.txt

  daily_papers/
    __init__.py
    hf_client.py          # Hugging Face Daily Papers API 클라이언트
    models.py             # 데이터 모델 (Paper, PaperSummary 등)
    pdf_downloader.py     # PDF 다운로드 및 캐시
    pdf_parser.py         # PDF 텍스트/figure 추출

  llm/
    __init__.py
    client.py             # LLM 공통 클라이언트 (OpenAI 등)
    summarizer.py         # 논문 → 구조화 요약 JSON 생성 로직
    prompt_templates.py   # 프롬프트 템플릿

  slides/
    __init__.py
    markdown_builder.py   # Marp용 markdown 생성
    marp_renderer.py      # marp CLI 호출, 이미지 생성

  tts/
    __init__.py
    client.py             # TTS API 호출

  video/
    __init__.py
    builder.py            # moviepy로 이미지 + 오디오 → mp4

  youtube/
    __init__.py
    uploader.py           # YouTube Data API 업로드

  storage/
    __init__.py
    paths.py              # 날짜/논문별 폴더 구조 관리
    metadata_store.py     # (선택) CSV/JSON 등으로 메타데이터 저장

  logs/
    ...
  outputs/
    ...
```

---

## 4. 설정/환경 변수

`config.py` 또는 `.env`로 관리:

* `HF_BASE_URL` = `"https://huggingface.co"`
* `TOP_K` = 기본 10
* `OPENAI_API_KEY`
* `OPENAI_LLM_MODEL` (예: `"gpt-4o-mini"`)
* `OPENAI_TTS_MODEL` (예: `"gpt-4o-mini-tts"`)
* `OPENAI_TTS_VOICE` (예: `"alloy"`)
* `TTS_STYLE_INSTRUCTION` (옵션)
* `TTS_SPEED` (기본 1.2)
* `YOUTUBE_CLIENT_SECRETS_FILE` (OAuth용 JSON 경로)
* `YOUTUBE_TOKEN_FILE` (토큰 캐시 파일 경로)
* `OUTPUT_BASE_DIR` (예: `"./outputs"`)
* `LOG_LEVEL` (INFO/DEBUG)

---

## 5. 데이터 모델 상세

### 5.1. HF Daily Papers 응답 → 내부 모델

```python
# daily_papers/models.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HFPaperEntry:
    paper_id: str         # HF에서 제공하는 paper.id (arxiv id 등)
    title: str
    summary: str
    authors: List[str]
    upvotes: int
    published_at: str
    hf_url: Optional[str] = None
    arxiv_url: Optional[str] = None
    pdf_url: Optional[str] = None

@dataclass
class ExtractedText:
    abstract: str
    intro: str
    conclusion: str
    # 필요시 method, experiments 등 추가
```

### 5.2. LLM 요약 결과 모델

```python
# llm/summarizer.py

from dataclasses import dataclass
from typing import List

@dataclass
class SlideSpec:
    title: str
    bullets: List[str]
    script: str           # 이 슬라이드에서 말할 나레이션 텍스트
    figure_hint: str | None = None  # "Figure 1" 등 힌트(optional)

@dataclass
class PaperSummary:
    paper_id: str
    title: str
    category: str
    one_line: str
    key_ideas: List[str]
    insights: List[dict]  # {"label": str, "content": str}
    slides: List[SlideSpec]
```

### 5.3. 전체 데일리 에피소드 모델

```python
@dataclass
class DailyEpisode:
    date: str
    papers: List[PaperSummary]
```

---

## 6. 모듈별 상세 요구사항

### 6.1. `daily_papers/hf_client.py`

**기능**

1. `fetch_daily_papers(date: str, top_k: int) -> List[HFPaperEntry]`

   * `GET /api/daily_papers?date={date}` 호출.
   * 응답 JSON 파싱 후 `HFPaperEntry` 리스트로 변환.
   * `upvotes` 기준 내림차순 정렬 후 `top_k`만 반환.

2. (선택) `resolve_arxiv_and_pdf(entry: HFPaperEntry) -> HFPaperEntry`

   * HF Paper 페이지 URL이 있다면 HTML을 파싱해서

     * arXiv URL,
     * PDF URL
       를 추출.
   * 단순 MVP에서는 `https://arxiv.org/pdf/{paper_id}.pdf` 형식으로 가정 가능(필요시 실패 시 fallback 로직).

### 6.2. `daily_papers/pdf_downloader.py`

**기능**

* `download_pdf(paper: HFPaperEntry, date: str) -> str`

  * `OUTPUT_BASE_DIR/date/{paper_id}/paper.pdf` 위치에 저장.
  * 이미 존재하면 다운로드 스킵.

### 6.3. `daily_papers/pdf_parser.py`

**기능**

* `extract_core_text(pdf_path: str) -> ExtractedText`

  * `pymupdf`로 PDF 열기.
  * 각 페이지의 텍스트를 읽어

    * “Abstract”, “Introduction”, “Conclusion(s)” 등의 키워드 기반으로 구간 스니펫 추출 (단순 heuristic 가능).
  * `ExtractedText` 리턴.

(고급 기능: figure 이미지를 추출해 `figures/` 폴더에 저장. MVP에선 생략 가능.)

---

### 6.4. `llm/client.py`

**기능**

* OpenAI LLM 호출 래퍼:

  * `generate_json(prompt: str, schema: dict) -> dict`
  * 에러 시 재시도, rate limit 처리 등 기본적인 예외 처리 포함.

### 6.5. `llm/prompt_templates.py`

**내용**

* 논문 요약용 프롬프트 템플릿 문자열:

  * system: “당신은 AI 논문을 유튜브 발표용으로 구조화해 주는 전문가다…”
  * user:

    * 논문 메타데이터(제목, 저자, 요약, 링크 등) +
    * `ExtractedText` (abstract, intro, conclusion의 텍스트)
    * * 출력 JSON 스키마 설명
  * 예시 JSON 스키마:

    ```json
    {
      "paper_id": "string",
      "title": "string",
      "category": "string",
      "one_line": "string",
      "key_ideas": ["string", "string"],
      "insights": [
        { "label": "Why it matters", "content": "..." },
        { "label": "Limitations", "content": "..." }
      ],
      "slides": [
        {
          "title": "string",
          "bullets": ["string", "string"],
          "script": "슬라이드 전체를 설명하는 자연스러운 말",
          "figure_hint": "optional"
        }
      ]
    }
    ```

### 6.6. `llm/summarizer.py`

**기능**

* `summarize_paper(paper: HFPaperEntry, text: ExtractedText) -> PaperSummary`

  * 위 프롬프트 템플릿을 사용해 LLM 호출.
  * 응답 JSON을 `PaperSummary` 데이터 클래스로 파싱.
  * 오류/형식 이상 시 재시도 또는 fallback(간단 요약만 생성).

---

### 6.7. `slides/markdown_builder.py`

**기능**

* `build_daily_markdown(daily: DailyEpisode, out_path: str) -> None`

  * Marp 포맷의 markdown 파일 생성.

  * 구조 예시:

    ```markdown
    ---
    marp: true
    title: "Daily AI Papers - {{date}}"
    paginate: true
    ---

    # 오늘의 Top {{N}} AI Papers ({{date}})

    - Source: Hugging Face Daily Papers
    - Created automatically

    ---

    <!-- 논문 1 시작 -->
    # [1] {{paper.title}}

    - 한 줄 요약: {{paper.one_line}}
    - 분야: {{paper.category}}

    ---

    # [1] Key Ideas

    - {{paper.key_ideas[0]}}
    - {{paper.key_ideas[1]}}

    ---

    # [1] Slide Title from LLM

    - bullet 1
    - bullet 2
    ```

  * 슬라이드 번호 관리: 논문 순서 + 슬라이드 인덱스를 사용해 내부적으로 tracking.

---

### 6.8. `slides/marp_renderer.py`

**기능**

* `render_markdown_to_images(md_path: str, output_prefix: str) -> List[str]`

  * 내부적으로 shell 명령 호출:

    ```bash
    marp {md_path} --images {output_prefix}
    ```

  * 결과 파일명 패턴:

    * `{output_prefix}001.png`, `{output_prefix}002.png`, …

  * 반환값: 이미지 파일 경로 리스트(정렬된 순서).

---

### 6.9. `tts/client.py`

**기능**

* `synthesize(text: str, out_path: str) -> None`

  * 설정된 TTS API(OpenAI 등)로 텍스트 → 오디오 파일 생성.
  * 형식: mp3 또는 wav.

* `synthesize_for_slides(slides: List[SlideSpec], base_dir: str) -> List[str]`

  * 슬라이드 리스트를 받아

    * `audio_slide_001.mp3`, `audio_slide_002.mp3` … 순서대로 생성.
  * 반환: 오디오 파일 경로 리스트 (슬라이드 순서와 동일한 인덱스).

---

### 6.10. `video/builder.py`

**핵심 포인트: 오디오-슬라이드 싱크**

* **슬라이드별로 이미 나눠진 오디오**를 사용한다.
* 각 슬라이드 이미지의 `duration`을 **해당 오디오 길이**에 맞춘다.
* 이 방식을 통해,

  * 오디오가 끝나는 타이밍에 정확히 **다음 슬라이드로 전환**되도록 한다.

**기능**

* `build_video(slide_images: List[str], audio_files: List[str], out_path: str) -> None`

  * 조건:

    * `len(slide_images) == len(audio_files)` 가정.
  * 로직(의사코드):

    ```python
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    def build_video(slide_images, audio_files, out_path):
        clips = []
        for img, audio in zip(slide_images, audio_files):
            a = AudioFileClip(audio)
            c = ImageClip(img).set_duration(a.duration)
            c = c.set_audio(a)
            clips.append(c)

        final = concatenate_videoclips(clips)
        final.write_videofile(out_path, fps=30)
    ```

---

### 6.11. `youtube/uploader.py`

**기능**

1. `get_youtube_client()`

   * OAuth 인증 처리 후 YouTube API 클라이언트 생성.
   * `client_secrets.json`, 토큰 캐시 파일 사용.

2. `upload_video(video_path: str, title: str, description: str, tags: List[str]) -> str`

   * `videos.insert` 호출.
   * 성공 시 `videoId` 반환.

3. 제목/설명 템플릿 예시

   * 제목:

     * `"Daily AI Papers - {date} | Top {N} ML Papers"`
   * 설명:

     * 각 논문 리스트(제목, 저자, 링크)
     * “Source: Hugging Face Daily Papers, original authors” 크레딧.

---

## 7. `main.py` 동작 요구사항

**CLI 인터페이스**

```bash
python main.py --date 2025-11-22 --top-k 10
```

* 옵션

  * `--date`: 선택, 없으면 오늘 날짜.
  * `--top-k`: 선택, 없으면 config의 기본값.

**로직**

1. 파라미터 파싱.
2. HF Daily Papers에서 Top K 논문 리스트 가져오기.
3. 각 논문에 대해:

   * PDF 다운로드
   * 텍스트 추출
   * LLM 요약(`PaperSummary`) 생성
4. `DailyEpisode` 객체 생성.
5. 마크다운 생성 → 이미지 렌더링.
6. 슬라이드별 TTS 생성.
7. 비디오 생성.
8. YouTube 업로드.
9. 로그 출력 및 종료.

에러 처리:

* 논문 하나 처리 실패 시:

  * 로그만 남기고 나머지 논문은 계속 진행.
* LLM/TTS/YouTube API 오류 시:

  * 최대 N번 재시도 후 실패 처리.
  * 전체 파이프라인 실패 시 exit code 1.

---

## 8. 향후 확장 포인트(참고용)

* Shorts 버전(논문당 1개 짧은 영상) 추가.
* 썸네일 자동 생성 (LLM + 이미지 생성 또는 템플릿).
* figure 자동 추출 및 재그리기.
* LangGraph/MCP 기반 “에이전트 워크플로우”로 리팩터링.
