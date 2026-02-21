#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoML-like ablation (backward elimination) for agentic prompting-only MT pipeline
using OpenAI gpt-4o via LangChain ChatOpenAI.

Commands:
  - search: start from full pipeline, remove one component at a time if it improves metric
  - run: run translation with a given config and write output JSONL with "prediction"
  - eval: compute sacreBLEU on an output file that already has "prediction"

Notes:
- NEVER leak references (target/original_target) into prompts.
- Offline evaluation uses target/original_target only after predictions are produced.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sacrebleu.metrics import BLEU, CHRF


# =========================
# IO
# =========================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# Helpers
# =========================

def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _extract_json_substring(text: str) -> str:
    t = _strip_code_fences(text)
    m = re.search(r"[\{\[]", t)
    if not m:
        return t
    start = m.start()
    end_obj = t.rfind("}")
    end_arr = t.rfind("]")
    end = max(end_obj, end_arr)
    if end == -1 or end <= start:
        return t[start:]
    return t[start:end + 1]

def _safe_json_loads(text: str) -> Any:
    s = _extract_json_substring(text)
    return json.loads(s)

def _pretty_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)

def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(12.0, 0.7 * (2 ** attempt)))

def clean_translation(text: str) -> str:
    t = _strip_code_fences(text)
    for prefix in ["Korean:", "한국어:", "번역:", "번역문:", "KO:"]:
        if t.startswith(prefix):
            t = t[len(prefix):].lstrip()
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()
    return t

def as_float_safe(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# =========================
# Deterministic constraint helpers
# =========================

_URL_RE = re.compile(r"""https?://[^\s]+""", re.IGNORECASE)
_EMAIL_RE = re.compile(r"""[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}""")
_HANDLE_RE = re.compile(r"""[@#][A-Za-z0-9_]+""")
_CODETOKEN_RE = re.compile(r"""`[^`]+`""")
_NUM_RE = re.compile(r"""(?<!\w)(\d[\d,]*(?:\.\d+)?)(?!\w)""")

def extract_auto_do_not_translate(source: str) -> List[str]:
    toks = set()
    for m in _URL_RE.finditer(source or ""):
        toks.add(m.group(0))
    for m in _EMAIL_RE.finditer(source or ""):
        toks.add(m.group(0))
    for m in _HANDLE_RE.finditer(source or ""):
        toks.add(m.group(0))
    for m in _CODETOKEN_RE.finditer(source or ""):
        toks.add(m.group(0))
    for tok in re.findall(r"""(?<!\w)(?:--?[A-Za-z0-9_-]+)(?!\w)""", source or ""):
        if len(tok) >= 2:
            toks.add(tok)
    return sorted(toks)

def digit_numbers_in_text(s: str) -> List[str]:
    return [m.group(1) for m in _NUM_RE.finditer(s or "")]

def check_numbers_preserved(source: str, target: str) -> Tuple[bool, List[str]]:
    src_nums = digit_numbers_in_text(source)
    missing = []
    for n in src_nums:
        if n not in (target or ""):
            missing.append(n)
    return (len(missing) == 0, missing)

def check_newlines_preserved(source: str, target: str) -> bool:
    return (source or "").count("\n") == (target or "").count("\n")


# =========================
# Prompts
# =========================

STYLEPLANNER_SYSTEM = """\
You are StylePlanner, an expert in English→Korean (South Korea) translation style control.
Your job: infer the best Korean register/style for the given source and domain, and output a compact JSON style spec.
You MUST NOT translate the sentence. Only produce JSON.
Be conservative: choose a style that matches professional human translation for the domain.
"""

STYLEPLANNER_USER = Template("""\
[Task]
Decide the optimal Korean style/register for translating the SOURCE into Korean (ko-KR).
Return JSON only.

[Metadata]
lp: en-ko_KR
domain: $DOMAIN
document_id: $DOC_ID
segment_id: $SEG_ID

[Document memory (may be empty)]
preferred_style_so_far: $MEM_STYLE

[SOURCE]
$SOURCE

[Output JSON schema]
{
  "speech_level": "plain|formal_polite|polite|intimate|mixed",
  "register": "news|social|speech|literary|other",
  "narration_style": "written|spoken",
  "quote_style": "double_quotes|single_quotes|korean_quotes",
  "proper_noun_policy": "transliterate_with_parens|keep_original|mixed",
  "sentence_endings_policy": "keep_source_tone|normalize_to_register",
  "formatting_rules": ["rule1","rule2"],
  "notes_for_translator": ["short actionable instructions"]
}
""")

TERMEXTRACT_SYSTEM = """\
You are TermExtractor for English→Korean translation.
Goal: extract named entities, technical terms, ambiguous terms, numbers/dates/units, and propose a consistent Korean rendering.
Respect document-level consistency: if a term already exists in the provided glossary, keep it unless it is clearly wrong.
Return JSON only. Do NOT translate the whole sentence.
"""

TERMEXTRACT_USER = Template("""\
[Task]
Build/extend a glossary and constraints for translating SOURCE into Korean (ko-KR).
Return JSON only.

[Inputs]
domain: $DOMAIN
style_spec: $STYLE_SPEC_JSON

[Document glossary memory (JSON map: English_term -> Korean_term)]
$MEM_GLOSSARY_JSON

[Auto do-not-translate tokens (must be preserved exactly)]
$AUTO_DNT

[SOURCE]
$SOURCE

[Output JSON schema]
{
  "glossary_add": [
    {"src":"string","tgt":"string","type":"PERSON|ORG|LOC|TERM|ACRONYM|TITLE|OTHER","confidence":0.0,"notes":"short"}
  ],
  "do_not_translate": ["token1","token2"],
  "numbers_units": [
    {"src":"string","tgt_format":"string","notes":"keep digits/commas/unit"}
  ],
  "ambiguities_to_watch": [
    {"src_fragment":"string","possible_senses":["sense1","sense2"],"preferred_ko":"string","notes":"short"}
  ],
  "consistency_rules": ["short rule"]
}
""")

TRANSLATOR_SYSTEM = """\
You are a professional translator from English to Korean (South Korea).

Hard requirements:
- Output ONLY the Korean translation text. No explanations, no headings, no JSON.
- Preserve meaning exactly. Do not add new information.
- Preserve all numbers, dates, names, symbols, and formatting (including newlines).
- Follow the provided style_spec and glossary strictly.
- Tokens listed in do_not_translate MUST appear exactly unchanged.
"""

TRANSLATOR_USER = Template("""\
[Mode] $MODE
$MODE_INSTRUCTIONS

[Metadata]
domain: $DOMAIN
document_id: $DOC_ID
segment_id: $SEG_ID

[Style spec]
$STYLE_SPEC_JSON

[Glossary & constraints]
$GLOSSARY_JSON

[Document context (optional, brief)]
$DOC_CONTEXT

[SOURCE]
$SOURCE

[Output]
Korean translation only.
""")

ADEQUACY_SYSTEM = """\
You are AdequacyReviewer for English→Korean translation.
Find meaning errors: omissions, additions, wrong relations, wrong polarity/negation, wrong numbers/dates, wrong named entities, mistranslations.
Return a strict JSON report only.
Do NOT rewrite the full translation.
"""

ADEQUACY_USER = Template("""\
[Task]
Evaluate adequacy of the Korean translation vs the English source.
Return JSON only.

[Metadata]
domain: $DOMAIN
style_spec: $STYLE_SPEC_JSON

[SOURCE]
$SOURCE

[TRANSLATION]
$CANDIDATE_KO

[Output JSON schema]
{
  "score_0_100": 0,
  "critical": false,
  "issues": [
    {
      "category":"OMISSION|ADDITION|MISTRANSLATION|NUMBER|NAME|NEGATION|COREFERENCE|OTHER",
      "severity":"major|minor",
      "src_evidence":"short quote",
      "tgt_evidence":"short quote",
      "fix_instruction":"one-line actionable fix",
      "suggested_rewrite_fragment":"optional short Korean fragment"
    }
  ]
}
""")

FLUENCY_SYSTEM = """\
You are FluencyStyleReviewer for Korean (South Korea).
Improve grammar/spacing/punctuation/register consistency WITHOUT changing meaning.
Return a strict JSON report only.
Do NOT rewrite the full translation.
"""

FLUENCY_USER = Template("""\
[Task]
Review fluency/style of the Korean translation under the given style_spec.
Return JSON only.

[style_spec]
$STYLE_SPEC_JSON

[SOURCE for meaning constraints]
$SOURCE

[TRANSLATION]
$CANDIDATE_KO

[Output JSON schema]
{
  "score_0_100": 0,
  "issues": [
    {
      "category":"GRAMMAR|SPACING|PUNCTUATION|REGISTER|WORDCHOICE|REDUNDANCY|OTHER",
      "severity":"major|minor",
      "tgt_evidence":"short quote",
      "fix_instruction":"one-line actionable fix",
      "suggested_rewrite_fragment":"optional short Korean fragment"
    }
  ]
}
""")

SYNTH_SYSTEM = """\
You are FeedbackSynthesizer.
Input: two JSON review reports (adequacy + fluency/style) and glossary constraints.
Output: a single prioritized fix list that is non-redundant and easy for an editor to apply.
Return JSON only.
"""

SYNTH_USER = Template("""\
[Inputs]
style_spec: $STYLE_SPEC_JSON
glossary: $GLOSSARY_JSON

[Adequacy report JSON]
$ADEQUACY_JSON

[Fluency/style report JSON]
$FLUENCY_JSON

[Output JSON schema]
{
  "priority_fixes": [
    {"priority": 1, "type":"ADEQUACY|FLUENCY|FORMAT|CONSISTENCY",
     "instruction":"single actionable instruction", "must_follow_glossary": true}
  ],
  "optional_polish": ["short instruction"]
}
""")

EDITOR_SYSTEM = """\
You are PostEditEditor for English→Korean translation.
Apply the prioritized fix list with minimal edits.
Never introduce new meaning.
Obey glossary and style_spec strictly.
Preserve newlines exactly.
Output ONLY the corrected Korean translation.
"""

EDITOR_USER = Template("""\
[Metadata]
domain: $DOMAIN
style_spec: $STYLE_SPEC_JSON
glossary: $GLOSSARY_JSON

[SOURCE]
$SOURCE

[CURRENT TRANSLATION]
$CANDIDATE_KO

[PRIORITY FIXES JSON]
$FIXLIST_JSON

[Output]
Corrected Korean translation only.
""")

EDITORLITE_SYSTEM = """\
You are PostEditEditor for English→Korean translation.
Improve translation while preserving meaning exactly.
Fix obvious adequacy issues (numbers/names/negation/omissions) and fluency.
Obey glossary/style_spec. Preserve newlines. Output ONLY corrected Korean.
"""

EDITORLITE_USER = Template("""\
[Metadata]
domain: $DOMAIN
style_spec: $STYLE_SPEC_JSON
glossary: $GLOSSARY_JSON

[SOURCE]
$SOURCE

[CURRENT TRANSLATION]
$CANDIDATE_KO

[Output]
Corrected Korean translation only.
""")

PROOF_SYSTEM = """\
You are a Korean proofreader (ko-KR).
Only fix spacing/punctuation/typos/register consistency.
Do NOT change meaning/terms/names/numbers unless obvious typo.
Preserve newlines exactly.
Output ONLY final Korean text.
"""

PROOF_USER = Template("""\
[style_spec]
$STYLE_SPEC_JSON

[SOURCE (for meaning constraints)]
$SOURCE

[TEXT TO PROOFREAD]
$EDITED_KO

[Output]
Final Korean text only.
""")

GATE_SYSTEM = """\
You are Gatekeeper for translation outputs.
Check hard constraints:
- Output contains translation text only
- All digit-based numbers from source are preserved
- Glossary terms respected as much as possible
- Newline structure preserved
Return JSON only with pass/fail and minimal fix instructions.
"""

GATE_USER = Template("""\
[style_spec]
$STYLE_SPEC_JSON

[glossary]
$GLOSSARY_JSON

[SOURCE]
$SOURCE

[FINAL KOREAN]
$FINAL_KO

[Output JSON schema]
{
  "pass": true,
  "violations": [
    {"type":"FORMAT|NUMBER|GLOSSARY|EXTRA_TEXT|OMISSION|OTHER","evidence":"short","fix_instruction":"one-line"}
  ]
}
""")


# =========================
# LLM wrapper (ChatOpenAI)
# =========================

@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    timeout: float = 180.0

    temperature_text: float = 0.0
    temperature_json: float = 0.0

    max_completion_tokens_text: int = 2400
    max_completion_tokens_json: int = 900

    max_retries: int = 4


def _mk_chat_openai(
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    temperature: float,
    max_completion_tokens: int,
    response_format: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> ChatOpenAI:
    # Use model_kwargs for max_completion_tokens / response_format (flattened to payload)
    model_kwargs: Dict[str, Any] = {"max_completion_tokens": max_completion_tokens}
    if response_format is not None:
        model_kwargs["response_format"] = response_format

    # LangChain minor version compat
    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            max_retries=max_retries,
            model_kwargs=model_kwargs,
        )
    except TypeError:
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            request_timeout=timeout,
            temperature=temperature,
            max_retries=max_retries,
            model_kwargs=model_kwargs,
        )


class LLMCaller:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.llm_text = _mk_chat_openai(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            temperature=cfg.temperature_text,
            max_completion_tokens=cfg.max_completion_tokens_text,
            response_format=None,
            max_retries=cfg.max_retries,
        )
        # Try JSON mode; if model/provider doesn't like it, we fallback in call_json
        self.llm_json = _mk_chat_openai(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            temperature=cfg.temperature_json,
            max_completion_tokens=cfg.max_completion_tokens_json,
            response_format={"type": "json_object"},
            max_retries=cfg.max_retries,
        )

    def call_text(self, system: str, user: str, retries: int = 3) -> str:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.llm_text.invoke(msgs)
                return (resp.content or "").strip()
            except Exception as e:
                last_err = e
                _sleep_backoff(attempt)
        raise RuntimeError(f"LLM call_text failed: {last_err}")

    def call_json(self, system: str, user: str, retries: int = 3) -> Any:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.llm_json.invoke(msgs)
                raw = (resp.content or "").strip()
                return _safe_json_loads(raw)
            except Exception as e:
                last_err = e
                _sleep_backoff(attempt)

        # Fallback: ask for JSON-only without response_format
        fix_system = "Return VALID JSON only. No markdown."
        fix_user = f"{user}\n\n(IMPORTANT: Output must be valid JSON only.)"
        raw2 = self.call_text(fix_system, fix_user, retries=2)
        return _safe_json_loads(raw2)


# =========================
# Pipeline config + doc memory
# =========================

@dataclass(frozen=True)
class PipelineConfig:
    use_styleplanner: bool = True
    use_term_extractor: bool = True
    use_doc_context: bool = True
    use_best_of_n: bool = True
    use_reviewers: bool = True
    use_editor: bool = True
    use_proofreader: bool = True
    use_gatekeeper: bool = True
    use_repair_loop: bool = True

    single_mode_when_no_bestof: str = "REFERENCE_LIKE"  # NATURAL / FAITHFUL / REFERENCE_LIKE

    def normalized(self) -> "PipelineConfig":
        if self.use_gatekeeper is False and self.use_repair_loop is True:
            return dataclasses.replace(self, use_repair_loop=False)
        return self

    def id_str(self) -> str:
        c = self.normalized()
        bits = [
            f"S{int(c.use_styleplanner)}",
            f"G{int(c.use_term_extractor)}",
            f"C{int(c.use_doc_context)}",
            f"B{int(c.use_best_of_n)}",
            f"R{int(c.use_reviewers)}",
            f"E{int(c.use_editor)}",
            f"P{int(c.use_proofreader)}",
            f"K{int(c.use_gatekeeper)}",
            f"L{int(c.use_repair_loop)}",
            f"smode={c.single_mode_when_no_bestof}",
        ]
        s = "|".join(bits)
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        return f"{'-'.join(bits[:9])}__{h}"

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self.normalized())

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "PipelineConfig":
        return PipelineConfig(**d).normalized()


@dataclass
class DocMemory:
    style_spec: Optional[Dict[str, Any]] = None
    glossary: Dict[str, str] = field(default_factory=dict)
    recent_context: List[Tuple[str, str]] = field(default_factory=list)

    def add_context(self, source: str, translation: str, keep_last: int = 2) -> None:
        self.recent_context.append((source, translation))
        if len(self.recent_context) > keep_last:
            self.recent_context = self.recent_context[-keep_last:]


# =========================
# Translator (configurable)
# =========================

class AgenticTranslator:
    def __init__(self, llm: LLMCaller):
        self.llm = llm
        self.doc_mem: Dict[str, DocMemory] = {}

    def _get_mem(self, doc_id: str) -> DocMemory:
        if doc_id not in self.doc_mem:
            self.doc_mem[doc_id] = DocMemory()
        return self.doc_mem[doc_id]

    def _render(self, tpl: Template, **kwargs: Any) -> str:
        cooked: Dict[str, str] = {}
        for k, v in kwargs.items():
            if isinstance(v, (dict, list)):
                cooked[k] = _pretty_json(v)
            elif v is None:
                cooked[k] = ""
            else:
                cooked[k] = str(v)
        return tpl.safe_substitute(**cooked)

    def default_style_spec(self, domain: str) -> Dict[str, Any]:
        reg = "news" if (domain or "").lower() == "news" else "other"
        return {
            "speech_level": "plain",
            "register": reg,
            "narration_style": "written",
            "quote_style": "korean_quotes",
            "proper_noun_policy": "mixed",
            "sentence_endings_policy": "normalize_to_register",
            "formatting_rules": ["preserve newlines exactly", "keep punctuation consistent"],
            "notes_for_translator": ["Do not add any information.", "Preserve numbers and named entities."],
        }

    def style_planner(self, cfg: PipelineConfig, domain: str, doc_id: str, seg_id: str, source: str) -> Dict[str, Any]:
        mem = self._get_mem(doc_id)
        if not cfg.use_styleplanner:
            if mem.style_spec is None:
                mem.style_spec = self.default_style_spec(domain)
            return mem.style_spec

        if mem.style_spec is not None:
            return mem.style_spec

        user = self._render(
            STYLEPLANNER_USER,
            DOMAIN=domain,
            DOC_ID=doc_id,
            SEG_ID=seg_id,
            MEM_STYLE=_pretty_json(mem.style_spec) if mem.style_spec else "",
            SOURCE=source,
        )
        style_spec = self.llm.call_json(STYLEPLANNER_SYSTEM, user)
        if not isinstance(style_spec, dict):
            style_spec = self.default_style_spec(domain)

        base = self.default_style_spec(domain)
        for k, v in base.items():
            style_spec.setdefault(k, v)

        mem.style_spec = style_spec
        return style_spec

    def term_extractor(
        self,
        cfg: PipelineConfig,
        domain: str,
        doc_id: str,
        seg_id: str,
        source: str,
        style_spec: Dict[str, Any],
        auto_dnt: List[str],
    ) -> Dict[str, Any]:
        mem = self._get_mem(doc_id)

        if not cfg.use_term_extractor:
            return {
                "glossary_add": [],
                "do_not_translate": list(auto_dnt),
                "numbers_units": [],
                "ambiguities_to_watch": [],
                "consistency_rules": [],
            }

        user = self._render(
            TERMEXTRACT_USER,
            DOMAIN=domain,
            STYLE_SPEC_JSON=style_spec,
            MEM_GLOSSARY_JSON=mem.glossary,
            AUTO_DNT=auto_dnt,
            SOURCE=source,
        )
        term_info = self.llm.call_json(TERMEXTRACT_SYSTEM, user)
        if not isinstance(term_info, dict):
            term_info = {}

        for item in term_info.get("glossary_add", []) or []:
            if not isinstance(item, dict):
                continue
            src = str(item.get("src", "")).strip()
            tgt = str(item.get("tgt", "")).strip()
            conf = as_float_safe(item.get("confidence", 0.0), 0.0)
            if src and tgt and conf >= 0.55:
                if src not in mem.glossary or not mem.glossary[src].strip():
                    mem.glossary[src] = tgt

        term_info.setdefault("do_not_translate", [])
        term_info["do_not_translate"] = sorted(set(term_info["do_not_translate"]) | set(auto_dnt))
        term_info.setdefault("numbers_units", [])
        term_info.setdefault("ambiguities_to_watch", [])
        term_info.setdefault("consistency_rules", [])
        return term_info

    def build_glossary_payload(self, doc_id: str, term_info: Dict[str, Any]) -> Dict[str, Any]:
        mem = self._get_mem(doc_id)
        return {
            "glossary": mem.glossary,
            "do_not_translate": term_info.get("do_not_translate") or [],
            "numbers_units": term_info.get("numbers_units") or [],
            "ambiguities_to_watch": term_info.get("ambiguities_to_watch") or [],
            "consistency_rules": term_info.get("consistency_rules") or [],
        }

    def doc_context_text(self, cfg: PipelineConfig, doc_id: str, max_chars: int = 900) -> str:
        if not cfg.use_doc_context:
            return ""
        mem = self._get_mem(doc_id)
        if not mem.recent_context:
            return ""
        lines = ["Previous segments (source -> translation):"]
        for i, (src, tgt) in enumerate(mem.recent_context[-2:], start=1):
            src_s = (src[:280] + "…") if len(src) > 280 else src
            tgt_s = (tgt[:280] + "…") if len(tgt) > 280 else tgt
            lines.append(f"{i}) EN: {src_s}")
            lines.append(f"   KO: {tgt_s}")
        out = "\n".join(lines)
        return out[:max_chars] + ("…" if len(out) > max_chars else "")

    def _translate_one_mode(
        self,
        domain: str,
        doc_id: str,
        seg_id: str,
        source: str,
        style_spec: Dict[str, Any],
        glossary_payload: Dict[str, Any],
        doc_ctx: str,
        mode: str,
        mode_instructions: str,
    ) -> str:
        user = self._render(
            TRANSLATOR_USER,
            MODE=mode,
            MODE_INSTRUCTIONS=mode_instructions,
            DOMAIN=domain,
            DOC_ID=doc_id,
            SEG_ID=seg_id,
            STYLE_SPEC_JSON=style_spec,
            GLOSSARY_JSON=glossary_payload,
            DOC_CONTEXT=doc_ctx,
            SOURCE=source,
        )
        out = self.llm.call_text(TRANSLATOR_SYSTEM, user)
        return clean_translation(out)

    def generate_candidates(
        self,
        cfg: PipelineConfig,
        domain: str,
        doc_id: str,
        seg_id: str,
        source: str,
        style_spec: Dict[str, Any],
        glossary_payload: Dict[str, Any],
    ) -> Dict[str, str]:
        doc_ctx = self.doc_context_text(cfg, doc_id)
        cands: Dict[str, str] = {}

        if not cfg.use_best_of_n:
            m = cfg.single_mode_when_no_bestof.upper()
            if m == "FAITHFUL":
                cands["FAITHFUL"] = self._translate_one_mode(
                    domain, doc_id, seg_id, source, style_spec, glossary_payload, doc_ctx,
                    "FAITHFUL",
                    "Translate as faithfully as possible. Prefer accuracy over elegance."
                )
            elif m == "NATURAL":
                cands["NATURAL"] = self._translate_one_mode(
                    domain, doc_id, seg_id, source, style_spec, glossary_payload, doc_ctx,
                    "NATURAL",
                    "Translate into natural, idiomatic Korean while preserving meaning exactly."
                )
            else:
                cands["REFERENCE_LIKE"] = self._translate_one_mode(
                    domain, doc_id, seg_id, source, style_spec, glossary_payload, doc_ctx,
                    "REFERENCE-LIKE",
                    "Translate into professional WMT-style written Korean for this domain."
                )
            return cands

        cands["FAITHFUL"] = self._translate_one_mode(
            domain, doc_id, seg_id, source, style_spec, glossary_payload, doc_ctx,
            "FAITHFUL",
            "Translate as faithfully as possible. Prefer accuracy over elegance."
        )
        cands["NATURAL"] = self._translate_one_mode(
            domain, doc_id, seg_id, source, style_spec, glossary_payload, doc_ctx,
            "NATURAL",
            "Translate into natural, idiomatic Korean while preserving meaning exactly."
        )
        cands["REFERENCE_LIKE"] = self._translate_one_mode(
            domain, doc_id, seg_id, source, style_spec, glossary_payload, doc_ctx,
            "REFERENCE-LIKE",
            "Translate into professional WMT-style written Korean for this domain."
        )
        return cands

    def review_candidate(
        self,
        domain: str,
        source: str,
        style_spec: Dict[str, Any],
        candidate_ko: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        adeq_user = self._render(
            ADEQUACY_USER,
            DOMAIN=domain,
            STYLE_SPEC_JSON=style_spec,
            SOURCE=source,
            CANDIDATE_KO=candidate_ko,
        )
        adeq = self.llm.call_json(ADEQUACY_SYSTEM, adeq_user)
        if not isinstance(adeq, dict):
            adeq = {}

        flu_user = self._render(
            FLUENCY_USER,
            STYLE_SPEC_JSON=style_spec,
            SOURCE=source,
            CANDIDATE_KO=candidate_ko,
        )
        flu = self.llm.call_json(FLUENCY_SYSTEM, flu_user)
        if not isinstance(flu, dict):
            flu = {}

        adeq_score = as_float_safe(adeq.get("score_0_100", 0), 0.0)
        flu_score = as_float_safe(flu.get("score_0_100", 0), 0.0)
        combined = 0.65 * adeq_score + 0.35 * flu_score

        if adeq.get("critical") is True:
            combined -= 15.0

        ok_nums, missing = check_numbers_preserved(source, candidate_ko)
        if not ok_nums:
            combined -= min(20.0, 5.0 * len(missing))
        if not check_newlines_preserved(source, candidate_ko):
            combined -= 5.0

        return adeq, flu, combined

    def synthesize_fixes(
        self,
        style_spec: Dict[str, Any],
        glossary_payload: Dict[str, Any],
        adeq: Dict[str, Any],
        flu: Dict[str, Any],
    ) -> Dict[str, Any]:
        user = self._render(
            SYNTH_USER,
            STYLE_SPEC_JSON=style_spec,
            GLOSSARY_JSON=glossary_payload,
            ADEQUACY_JSON=adeq,
            FLUENCY_JSON=flu,
        )
        fixlist = self.llm.call_json(SYNTH_SYSTEM, user)
        if not isinstance(fixlist, dict):
            fixlist = {"priority_fixes": [], "optional_polish": []}
        fixlist.setdefault("priority_fixes", [])
        fixlist.setdefault("optional_polish", [])
        return fixlist

    def edit_with_fixlist(
        self,
        domain: str,
        source: str,
        style_spec: Dict[str, Any],
        glossary_payload: Dict[str, Any],
        candidate_ko: str,
        fixlist: Dict[str, Any],
    ) -> str:
        user = self._render(
            EDITOR_USER,
            DOMAIN=domain,
            STYLE_SPEC_JSON=style_spec,
            GLOSSARY_JSON=glossary_payload,
            SOURCE=source,
            CANDIDATE_KO=candidate_ko,
            FIXLIST_JSON=fixlist,
        )
        out = self.llm.call_text(EDITOR_SYSTEM, user)
        return clean_translation(out)

    def edit_lite(
        self,
        domain: str,
        source: str,
        style_spec: Dict[str, Any],
        glossary_payload: Dict[str, Any],
        candidate_ko: str,
    ) -> str:
        user = self._render(
            EDITORLITE_USER,
            DOMAIN=domain,
            STYLE_SPEC_JSON=style_spec,
            GLOSSARY_JSON=glossary_payload,
            SOURCE=source,
            CANDIDATE_KO=candidate_ko,
        )
        out = self.llm.call_text(EDITORLITE_SYSTEM, user)
        return clean_translation(out)

    def proofread(self, source: str, style_spec: Dict[str, Any], edited_ko: str) -> str:
        user = self._render(
            PROOF_USER,
            STYLE_SPEC_JSON=style_spec,
            SOURCE=source,
            EDITED_KO=edited_ko,
        )
        out = self.llm.call_text(PROOF_SYSTEM, user)
        return clean_translation(out)

    def gatekeep(
        self,
        source: str,
        style_spec: Dict[str, Any],
        glossary_payload: Dict[str, Any],
        final_ko: str,
    ) -> Dict[str, Any]:
        user = self._render(
            GATE_USER,
            STYLE_SPEC_JSON=style_spec,
            GLOSSARY_JSON=glossary_payload,
            SOURCE=source,
            FINAL_KO=final_ko,
        )
        gate = self.llm.call_json(GATE_SYSTEM, user)
        if not isinstance(gate, dict):
            gate = {"pass": True, "violations": []}
        gate.setdefault("pass", True)
        gate.setdefault("violations", [])
        return gate

    def violations_to_fixlist(self, gate: Dict[str, Any]) -> Dict[str, Any]:
        fixes = []
        pr = 1
        for v in gate.get("violations") or []:
            if not isinstance(v, dict):
                continue
            instr = str(v.get("fix_instruction", "")).strip()
            if instr:
                fixes.append({"priority": pr, "type": "FORMAT", "instruction": instr, "must_follow_glossary": True})
                pr += 1
        return {"priority_fixes": fixes, "optional_polish": []}

    def select_candidate_without_review(self, source: str, candidates: Dict[str, str]) -> str:
        order_pref = {"REFERENCE_LIKE": 3, "FAITHFUL": 2, "NATURAL": 1}
        best_score = -1e9
        best_text = ""
        for k, cand in candidates.items():
            score = 0.0
            ok_nums, missing = check_numbers_preserved(source, cand)
            score += 10.0 if ok_nums else -5.0 * len(missing)
            score += 5.0 if check_newlines_preserved(source, cand) else -5.0
            score += order_pref.get(k, 0)
            if score > best_score:
                best_score = score
                best_text = cand
        return best_text or list(candidates.values())[0]

    def translate_one(self, cfg: PipelineConfig, row: Dict[str, Any]) -> str:
        cfg = cfg.normalized()

        domain = str(row.get("domain", "") or "unknown")
        doc_id = str(row.get("document_id", "") or row.get("doc_id", "") or "DOC_UNKNOWN")
        seg_id = str(row.get("segment_id", "") or row.get("seg_id", "") or "SEG_UNKNOWN")
        source = str(row.get("source", "") or row.get("src", "") or "")

        mem = self._get_mem(doc_id)

        style_spec = self.style_planner(cfg, domain, doc_id, seg_id, source)
        auto_dnt = extract_auto_do_not_translate(source)
        term_info = self.term_extractor(cfg, domain, doc_id, seg_id, source, style_spec, auto_dnt)
        glossary_payload = self.build_glossary_payload(doc_id, term_info)

        candidates = self.generate_candidates(cfg, domain, doc_id, seg_id, source, style_spec, glossary_payload)

        if cfg.use_reviewers:
            best_score = -1e9
            chosen = ""
            best_adeq, best_flu = {}, {}
            for cand in candidates.values():
                adeq, flu, score = self.review_candidate(domain, source, style_spec, cand)
                if score > best_score:
                    best_score = score
                    chosen = cand
                    best_adeq, best_flu = adeq, flu
            out = chosen
            if cfg.use_editor:
                fixlist = self.synthesize_fixes(style_spec, glossary_payload, best_adeq, best_flu)
                out = self.edit_with_fixlist(domain, source, style_spec, glossary_payload, out, fixlist)
        else:
            out = self.select_candidate_without_review(source, candidates)
            if cfg.use_editor:
                out = self.edit_lite(domain, source, style_spec, glossary_payload, out)

        if cfg.use_proofreader:
            out = self.proofread(source, style_spec, out)

        if cfg.use_gatekeeper:
            gate = self.gatekeep(source, style_spec, glossary_payload, out)
            if gate.get("pass") is False and cfg.use_repair_loop:
                fix2 = self.violations_to_fixlist(gate)
                out = self.edit_with_fixlist(domain, source, style_spec, glossary_payload, out, fix2)
                if cfg.use_proofreader:
                    out = self.proofread(source, style_spec, out)

        mem.add_context(source, out)
        return out


# =========================
# Eval & sampling
# =========================

def choose_reference(row: Dict[str, Any]) -> Optional[str]:
    t = row.get("target")
    if isinstance(t, str) and t.strip():
        return t
    ot = row.get("original_target")
    if isinstance(ot, str) and ot.strip():
        return ot
    return None

def filter_rows(rows: List[Dict[str, Any]], exclude_bad_source: bool) -> List[int]:
    idxs = []
    for i, r in enumerate(rows):
        if exclude_bad_source and bool(r.get("is_bad_source", False)):
            continue
        if choose_reference(r) is None:
            continue
        src = r.get("source") or r.get("src")
        if not isinstance(src, str) or not src.strip():
            continue
        idxs.append(i)
    return idxs

def group_by_doc(rows: List[Dict[str, Any]], idxs: List[int]) -> Dict[str, List[int]]:
    docs: Dict[str, List[int]] = {}
    for i in idxs:
        r = rows[i]
        doc_id = str(r.get("document_id", "") or r.get("doc_id", "") or "DOC_UNKNOWN")
        docs.setdefault(doc_id, []).append(i)
    return docs

def sample_by_docs(rows: List[Dict[str, Any]], idxs: List[int], sample_docs: int, seed: int) -> List[int]:
    docs = group_by_doc(rows, idxs)
    doc_ids = list(docs.keys())
    rnd = random.Random(seed)
    rnd.shuffle(doc_ids)
    chosen = doc_ids[:sample_docs] if sample_docs > 0 else doc_ids
    sampled = []
    for d in chosen:
        sampled.extend(docs[d])
    sampled.sort()
    return sampled

def compute_metric(metric: str, hyps: List[str], refs: List[str]) -> float:
    if metric.lower() == "bleu":
        # ✅ Korean BLEU 토크나이저 적용
        m = BLEU(tokenize="ko-mecab")
        return float(m.corpus_score(hyps, [refs]).score)

    # chrF는 문자 n-gram 기반이라 토크나이저 영향이 거의 없어서 그대로 두어도 됩니다.
    m = CHRF(word_order=2)  # chrF++
    return float(m.corpus_score(hyps, [refs]).score)


# =========================
# Cache
# =========================

def load_cached_predictions(run_dir: Path) -> Optional[Dict[int, str]]:
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        return None
    pred_map: Dict[int, str] = {}
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred_map[int(obj["idx"])] = obj["prediction"]
    return pred_map

def save_cached_predictions(run_dir: Path, pred_map: Dict[int, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for idx in sorted(pred_map.keys()):
            f.write(json.dumps({"idx": idx, "prediction": pred_map[idx]}, ensure_ascii=False) + "\n")


# =========================
# Run one config on subset
# =========================

def run_config_on_subset(
    llm_cfg: LLMConfig,
    cfg: PipelineConfig,
    rows: List[Dict[str, Any]],
    subset_idxs: List[int],
    cache_dir: Path,
    show_progress: bool = True,
) -> Dict[int, str]:
    cfg = cfg.normalized()
    run_dir = cache_dir / f"{cfg.id_str()}__{len(subset_idxs)}"
    cached = load_cached_predictions(run_dir)
    if cached is not None and all(i in cached for i in subset_idxs):
        return cached

    llm = LLMCaller(llm_cfg)
    pipe = AgenticTranslator(llm)
    pred_map: Dict[int, str] = cached or {}

    it = subset_idxs
    if show_progress:
        it = tqdm(subset_idxs, desc=f"Translating ({cfg.id_str()})", leave=False)

    for idx in it:
        if idx in pred_map:
            continue
        r = dict(rows[idx])

        # IMPORTANT: do not leak refs into prompts
        r.pop("target", None)
        r.pop("original_target", None)

        pred_map[idx] = pipe.translate_one(cfg, r)

    save_cached_predictions(run_dir, pred_map)
    return pred_map

def evaluate_config(
    llm_cfg: LLMConfig,
    cfg: PipelineConfig,
    rows: List[Dict[str, Any]],
    subset_idxs: List[int],
    metric: str,
    cache_dir: Path,
) -> float:
    pred_map = run_config_on_subset(llm_cfg, cfg, rows, subset_idxs, cache_dir, show_progress=True)
    hyps, refs = [], []
    for idx in subset_idxs:
        hyps.append(pred_map[idx])
        refs.append(choose_reference(rows[idx]) or "")
    return compute_metric(metric, hyps, refs)


# =========================
# Ablation search (backward elimination)
# =========================

ABLATION_FEATURES = [
    "use_styleplanner",
    "use_term_extractor",
    "use_doc_context",
    "use_best_of_n",
    "use_reviewers",
    "use_editor",
    "use_proofreader",
    "use_gatekeeper",
    "use_repair_loop",
]

def disable_feature(cfg: PipelineConfig, feat: str) -> PipelineConfig:
    if not hasattr(cfg, feat):
        raise ValueError(f"Unknown feature: {feat}")
    return dataclasses.replace(cfg, **{feat: False}).normalized()

def stepwise_backward_elimination(
    llm_cfg: LLMConfig,
    start_cfg: PipelineConfig,
    rows: List[Dict[str, Any]],
    subset_idxs: List[int],
    metric: str,
    cache_dir: Path,
    min_improve: float = 0.0,
    max_steps: int = 50,
) -> Tuple[PipelineConfig, float, List[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []

    cur_cfg = start_cfg.normalized()
    cur_score = evaluate_config(llm_cfg, cur_cfg, rows, subset_idxs, metric, cache_dir)
    history.append({"step": 0, "action": "start", "removed": "", "config_id": cur_cfg.id_str(), "score": cur_score})

    step = 0
    while step < max_steps:
        step += 1
        best_score = cur_score
        best_cfg = cur_cfg
        best_removed = ""

        for feat in ABLATION_FEATURES:
            if getattr(cur_cfg, feat, False) is False:
                continue
            cand_cfg = disable_feature(cur_cfg, feat)
            cand_score = evaluate_config(llm_cfg, cand_cfg, rows, subset_idxs, metric, cache_dir)
            history.append({"step": step, "action": "try_remove", "removed": feat, "config_id": cand_cfg.id_str(), "score": cand_score})
            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand_cfg
                best_removed = feat

        if best_cfg.id_str() != cur_cfg.id_str() and best_score >= cur_score + min_improve:
            cur_cfg = best_cfg
            cur_score = best_score
            history.append({"step": step, "action": "accept_remove", "removed": best_removed, "config_id": cur_cfg.id_str(), "score": cur_score})
        else:
            history.append({"step": step, "action": "stop", "removed": "", "config_id": cur_cfg.id_str(), "score": cur_score})
            break

    return cur_cfg, cur_score, history

def write_history_csv(path: Path, history: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "action", "removed", "config_id", "score"])
        for h in history:
            w.writerow([h["step"], h["action"], h["removed"], h["config_id"], h["score"]])


# =========================
# CLI
# =========================

def cmd_search(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    idxs = filter_rows(rows, exclude_bad_source=args.exclude_bad_source)
    subset_idxs = sample_by_docs(rows, idxs, sample_docs=args.sample_docs, seed=args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = LLMConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        temperature_text=args.temperature,
        temperature_json=0.0,
        max_completion_tokens_text=args.max_completion_tokens_text,
        max_completion_tokens_json=args.max_completion_tokens_json,
        max_retries=args.max_retries,
    )

    best_cfg, best_score, history = stepwise_backward_elimination(
        llm_cfg=llm_cfg,
        start_cfg=PipelineConfig().normalized(),
        rows=rows,
        subset_idxs=subset_idxs,
        metric=args.metric,
        cache_dir=cache_dir,
        min_improve=args.min_improve,
        max_steps=args.max_steps,
    )

    (cache_dir / "best_config.json").write_text(json.dumps(best_cfg.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_history_csv(cache_dir / "search_results.csv", history)

    print("=== DONE (search) ===")
    print(f"Metric: {args.metric}")
    print(f"Subset docs: {args.sample_docs}, rows: {len(subset_idxs)} (exclude_bad_source={args.exclude_bad_source})")
    print(f"Best score: {best_score:.4f}")
    print(f"Best config id: {best_cfg.id_str()}")
    print(f"Saved: {cache_dir / 'best_config.json'}")
    print(f"Saved: {cache_dir / 'search_results.csv'}")

def cmd_run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    idxs = filter_rows(rows, exclude_bad_source=args.exclude_bad_source)

    cfg = PipelineConfig().normalized()
    if args.config:
        cfg_json = json.loads(Path(args.config).read_text(encoding="utf-8"))
        cfg = PipelineConfig.from_json(cfg_json).normalized()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = LLMConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        temperature_text=args.temperature,
        temperature_json=0.0,
        max_completion_tokens_text=args.max_completion_tokens_text,
        max_completion_tokens_json=args.max_completion_tokens_json,
        max_retries=args.max_retries,
    )

    pred_map = run_config_on_subset(
        llm_cfg=llm_cfg,
        cfg=cfg,
        rows=rows,
        subset_idxs=idxs,
        cache_dir=cache_dir,
        show_progress=True,
    )

    out_rows = []
    for i, r in enumerate(rows):
        rr = dict(r)
        if i in pred_map:
            rr["prediction"] = pred_map[i]
        out_rows.append(rr)

    write_jsonl(args.output, out_rows)
    print("=== DONE (run) ===")
    print(f"Wrote: {args.output}")
    print(f"Config: {cfg.id_str()}")

def cmd_eval(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)

    ref_rows: List[Dict[str, Any]] = []
    ref_by_idx: Dict[int, Optional[str]] = {}
    ref_by_key: Dict[Tuple[Any, Any], Optional[str]] = {}
    bad_by_idx: Dict[int, bool] = {}

    if getattr(args, "ref", None):
        ref_rows = read_jsonl(args.ref)
        for pos, rr in enumerate(ref_rows):
            ref = choose_reference(rr)
            key_idx = rr.get("idx")
            if key_idx is not None:
                try:
                    ref_by_idx[int(key_idx)] = ref
                    bad_by_idx[int(key_idx)] = bool(rr.get("is_bad_source", False))
                except Exception:
                    pass
            ref_by_idx.setdefault(pos, ref)
            bad_by_idx.setdefault(pos, bool(rr.get("is_bad_source", False)))

            key = (rr.get("document_id"), rr.get("segment_id"))
            if key not in ref_by_key:
                ref_by_key[key] = ref

    hyps, refs = [], []
    skipped = 0
    for r in rows:
        if args.exclude_bad_source and bool(r.get("is_bad_source", False)):
            continue

        hyp = r.get("prediction", "")
        if not isinstance(hyp, str):
            hyp = ""

        ref = choose_reference(r)

        # Fallback to ref file: by idx, then by doc_id/segment_id, then by position if idx available
        if not isinstance(ref, str) or not ref.strip():
            idx = r.get("idx")
            ref_row = None
            if idx is not None:
                try:
                    idx_int = int(idx)
                    ref = ref_by_idx.get(idx_int, ref)
                    if ref_row is None and idx_int < len(ref_rows):
                        ref_row = ref_rows[idx_int]
                except Exception:
                    ref_row = None
            key = (r.get("document_id"), r.get("segment_id"))
            if (not isinstance(ref, str) or not ref.strip()) and ref_by_key:
                ref = ref_by_key.get(key, ref)
            if ref_row is None and idx is None and ref_rows:
                ref_row = ref_rows[min(len(ref_rows) - 1, len(hyps))]
            if ref_row and (not isinstance(ref, str) or not ref.strip()):
                ref = choose_reference(ref_row)

        # Exclude bad_source based on ref file if present
        if args.exclude_bad_source:
            idx = r.get("idx")
            if idx is not None:
                try:
                    if bad_by_idx.get(int(idx), False):
                        continue
                except Exception:
                    pass

        if not isinstance(ref, str) or not ref.strip():
            skipped += 1
            continue

        hyps.append(hyp)
        refs.append(ref)

    if not hyps:
        raise SystemExit("No hypotheses to score (prediction/target missing after filtering).")

    score = compute_metric(args.metric, hyps, refs)
    print(f"{args.metric.upper()} = {score:.4f}  (rows={len(hyps)}, skipped={skipped})")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser):
        p.add_argument("--input", required=True)
        p.add_argument("--exclude_bad_source", action="store_true")
        p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
        p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
        p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        p.add_argument("--timeout", type=float, default=180.0)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--max_completion_tokens_text", type=int, default=2400)
        p.add_argument("--max_completion_tokens_json", type=int, default=900)
        p.add_argument("--max_retries", type=int, default=4)

    p_search = sub.add_parser("search")
    add_common(p_search)
    p_search.add_argument("--metric", choices=["chrf", "bleu"], default="chrf")
    p_search.add_argument("--sample_docs", type=int, default=50)
    p_search.add_argument("--seed", type=int, default=13)
    p_search.add_argument("--cache_dir", default="runs_ablation_gpt4o")
    p_search.add_argument("--min_improve", type=float, default=0.0)
    p_search.add_argument("--max_steps", type=int, default=30)

    p_run = sub.add_parser("run")
    add_common(p_run)
    p_run.add_argument("--output", required=True)
    p_run.add_argument("--config", required=False)
    p_run.add_argument("--cache_dir", default="runs_cache_gpt4o")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--input", required=True)
    p_eval.add_argument("--metric", choices=["chrf", "bleu"], default="chrf")
    p_eval.add_argument("--exclude_bad_source", action="store_true")
    p_eval.add_argument("--ref", required=False, help="Optional reference JSONL (original data) if predictions file lacks targets.")

    args = ap.parse_args()

    if args.cmd in ("search", "run"):
        if not args.api_key:
            raise SystemExit("OPENAI_API_KEY is missing. Set env var or pass --api_key.")

    if args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
