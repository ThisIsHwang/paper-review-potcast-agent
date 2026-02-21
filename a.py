#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoML-like ablation (backward elimination) for an agentic prompting-only MT pipeline.
- vLLM (OpenAI-compatible) + LangChain ChatOpenAI
- Model: qwen3_30B-A3B-Instruct-2507
- Dataset: WMT24++ style JSONL (en-ko_KR.jsonl)
- IMPORTANT: Never leak references (target/original_target) into prompts.
- Offline evaluation only with sacrebleu (BLEU/chrF).

Subcommands:
  - search: start from full pipeline, remove one component at a time if it improves metric
  - run: run translation with a given config and write output JSONL with "prediction"
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
# Small utilities
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

def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
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
    return t[start:end+1]

def _safe_json_loads(text: str) -> Any:
    s = _extract_json_substring(text)
    return json.loads(s)

def _pretty_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)

def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(8.0, 0.5 * (2 ** attempt)))

def clean_translation(text: str) -> str:
    """Remove code fences and strip outer quotes if model returns them."""
    t = _strip_code_fences(text)
    # If the whole output is quoted, drop quotes
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()
    return t

def as_int_safe(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def as_float_safe(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# =========================
# Deterministic constraints helpers
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
# Prompt templates
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
- Think step-by-step internally, but do NOT output your reasoning.
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
You must find meaning errors: omissions, additions, wrong relations, wrong polarity/negation, wrong numbers/dates, wrong named entities, mistranslations.
Return a strict JSON report only (no prose outside JSON).
Do NOT rewrite the full translation.
Focus on high-impact issues first.
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
You must improve grammaticality, spacing, punctuation, register consistency, and naturalness WITHOUT changing meaning.
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
Apply the given prioritized fix list to the translation.

Rules:
- Minimal edits: change only what is necessary to address the fixes.
- Never introduce new meaning.
- Obey glossary and style_spec strictly.
- Preserve original newlines/paragraph structure exactly.
- Output ONLY the corrected Korean translation (no commentary).
- Think step-by-step internally, but do NOT output your reasoning.
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

Task:
- Improve the translation quality while preserving meaning exactly.
- Fix any obvious adequacy issues (numbers, names, negation, omissions) and fluency issues.
- Obey glossary/style_spec strictly.
- Preserve newlines exactly.
- Output ONLY the corrected Korean translation (no commentary).
- Think step-by-step internally, but do NOT output your reasoning.
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
Only fix spacing, punctuation, typos, and register consistency.
Do NOT change meaning, terminology, names, or numbers unless there is an obvious typo.
Preserve newlines exactly.
Output ONLY the final Korean text (no commentary).
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
- Output contains translation text only (no extra commentary/JSON)
- All digit-based numbers from source are preserved (no missing)
- Glossary terms are respected as much as possible
- Newline/paragraph structure preserved
Return JSON only with pass/fail and minimal fix instructions if fail.
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
# LLM wrapper
# =========================

@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    timeout: float = 180.0

    max_tokens_style: int = 512
    max_tokens_term: int = 768
    max_tokens_translation: int = 2048
    max_tokens_review: int = 1024
    max_tokens_synth: int = 512
    max_tokens_gate: int = 512


def create_chat_openai(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> ChatOpenAI:
    """
    Create ChatOpenAI against an OpenAI-compatible endpoint (vLLM).
    """
    try:
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except TypeError:
        # older versions fallback
        return ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=timeout,
        )


class LLMCaller:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._cache: Dict[Tuple[float, int], ChatOpenAI] = {}

    def _llm(self, temperature: float, max_tokens: int) -> ChatOpenAI:
        key = (temperature, max_tokens)
        if key not in self._cache:
            self._cache[key] = create_chat_openai(
                model=self.cfg.model,
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.cfg.timeout,
            )
        return self._cache[key]

    def call_text(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        retries: int = 3,
    ) -> str:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self._llm(temperature, max_tokens).invoke(msgs)
                return (resp.content or "").strip()
            except Exception as e:
                last_err = e
                _sleep_backoff(attempt)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def call_json(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        schema_hint: str,
        retries: int = 2,
    ) -> Any:
        raw = self.call_text(system, user, temperature=temperature, max_tokens=max_tokens, retries=3)
        try:
            return _safe_json_loads(raw)
        except Exception:
            # One repair call
            repair_system = "You are a JSON fixer. Return VALID JSON only. No markdown."
            repair_user = f"""Fix the following output into valid JSON only.

[Schema/requirements]
{schema_hint}

[Broken output]
{raw}
"""
            repaired = self.call_text(repair_system, repair_user, temperature=0.0, max_tokens=max_tokens, retries=retries)
            return _safe_json_loads(repaired)


# =========================
# Pipeline config + doc memory
# =========================

@dataclass(frozen=True)
class PipelineConfig:
    # Core switches (ablation targets)
    use_styleplanner: bool = True
    use_term_extractor: bool = True
    use_doc_context: bool = True
    use_best_of_n: bool = True
    use_reviewers: bool = True
    use_editor: bool = True
    use_proofreader: bool = True
    use_gatekeeper: bool = True
    use_repair_loop: bool = True

    # Candidate settings
    single_mode_when_no_bestof: str = "REFERENCE_LIKE"  # NATURAL / FAITHFUL / REFERENCE_LIKE

    # Temperatures per mode (kept fixed for fairness)
    temp_faithful: float = 0.3
    temp_natural: float = 0.7
    temp_reference: float = 0.5

    # Multi-sample per mode (optional; keep 1 by default to reduce noise/cost)
    n_faithful: int = 1
    n_natural: int = 1
    n_reference: int = 1

    # Reviewer weight
    w_adequacy: float = 0.65
    w_fluency: float = 0.35

    def normalized(self) -> "PipelineConfig":
        """
        Enforce dependency sanity:
        - repair_loop only meaningful if gatekeeper enabled
        """
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
        ]
        bits.append(f"smode={c.single_mode_when_no_bestof}")
        bits.append(f"tf={c.temp_faithful},tn={c.temp_natural},tr={c.temp_reference}")
        bits.append(f"nf={c.n_faithful},nn={c.n_natural},nr={c.n_reference}")
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
    glossary: Dict[str, str] = field(default_factory=dict)  # English_term -> Korean_term
    recent_context: List[Tuple[str, str]] = field(default_factory=list)

    def add_context(self, source: str, translation: str, keep_last: int = 2) -> None:
        self.recent_context.append((source, translation))
        if len(self.recent_context) > keep_last:
            self.recent_context = self.recent_context[-keep_last:]


# =========================
# Agentic translator (configurable)
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
        # Conservative defaults for WMT-ish human translation
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
        style_spec = self.llm.call_json(
            STYLEPLANNER_SYSTEM,
            user,
            temperature=0.0,
            max_tokens=self.llm.cfg.max_tokens_style,
            schema_hint="Must be a JSON object matching the provided schema exactly.",
        )
        if not isinstance(style_spec, dict):
            style_spec = self.default_style_spec(domain)

        # Fill defaults if missing
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
            # No new glossary; still enforce auto DNT
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
        term_info = self.llm.call_json(
            TERMEXTRACT_SYSTEM,
            user,
            temperature=0.0,
            max_tokens=self.llm.cfg.max_tokens_term,
            schema_hint="Return JSON with keys: glossary_add(list), do_not_translate(list), numbers_units(list), ambiguities_to_watch(list), consistency_rules(list).",
        )
        if not isinstance(term_info, dict):
            term_info = {}

        # Update doc glossary memory
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
        # Always include auto DNT tokens
        term_info["do_not_translate"] = sorted(set(term_info["do_not_translate"]) | set(auto_dnt))
        term_info.setdefault("numbers_units", [])
        term_info.setdefault("ambiguities_to_watch", [])
        term_info.setdefault("consistency_rules", [])
        return term_info

    def build_glossary_payload(
        self,
        doc_id: str,
        term_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        mem = self._get_mem(doc_id)
        payload = {
            "glossary": mem.glossary,
            "do_not_translate": term_info.get("do_not_translate") or [],
            "numbers_units": term_info.get("numbers_units") or [],
            "ambiguities_to_watch": term_info.get("ambiguities_to_watch") or [],
            "consistency_rules": term_info.get("consistency_rules") or [],
        }
        return payload

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

    def translate_mode(
        self,
        mode: str,
        mode_instructions: str,
        domain: str,
        doc_id: str,
        seg_id: str,
        source: str,
        style_spec: Dict[str, Any],
        glossary_payload: Dict[str, Any],
        temperature: float,
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
            DOC_CONTEXT=self.doc_context_text(PipelineConfig(), doc_id)  # placeholder; overridden below
            if False else "",
            SOURCE=source,
        )
        # Re-render with cfg-controlled context
        # (This keeps Template usage simple; we just string replace DOC_CONTEXT afterwards.)
        return user

    def _translate_call(
        self,
        user_prompt: str,
        temperature: float,
    ) -> str:
        out = self.llm.call_text(
            TRANSLATOR_SYSTEM,
            user_prompt,
            temperature=temperature,
            max_tokens=self.llm.cfg.max_tokens_translation,
            retries=3,
        )
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
        cfg = cfg.normalized()
        candidates: Dict[str, str] = {}

        # Context injection (optional)
        doc_ctx = self.doc_context_text(cfg, doc_id)

        def mk_user(mode: str, inst: str, style_spec: Dict[str, Any], glossary_payload: Dict[str, Any]) -> str:
            return self._render(
                TRANSLATOR_USER,
                MODE=mode,
                MODE_INSTRUCTIONS=inst,
                DOMAIN=domain,
                DOC_ID=doc_id,
                SEG_ID=seg_id,
                STYLE_SPEC_JSON=style_spec,
                GLOSSARY_JSON=glossary_payload,
                DOC_CONTEXT=doc_ctx,
                SOURCE=source,
            )

        # If no best-of-n: single candidate
        if not cfg.use_best_of_n:
            mode = cfg.single_mode_when_no_bestof.upper()
            if mode == "FAITHFUL":
                user = mk_user(
                    "FAITHFUL",
                    "Translate as faithfully as possible. Prefer accuracy over elegance. Keep structure close when possible.",
                    style_spec, glossary_payload
                )
                candidates["FAITHFUL"] = self._translate_call(user, cfg.temp_faithful)
            elif mode == "NATURAL":
                user = mk_user(
                    "NATURAL",
                    "Translate into natural, idiomatic Korean while preserving meaning exactly. Avoid awkward calques.",
                    style_spec, glossary_payload
                )
                candidates["NATURAL"] = self._translate_call(user, cfg.temp_natural)
            else:
                user = mk_user(
                    "REFERENCE-LIKE",
                    "Translate into professional WMT-style written Korean for this domain. Keep terms consistent and punctuation clean.",
                    style_spec, glossary_payload
                )
                candidates["REFERENCE_LIKE"] = self._translate_call(user, cfg.temp_reference)
            return candidates

        # Best-of-n: 3 modes (with optional multiple samples per mode)
        for i in range(cfg.n_faithful):
            key = f"FAITHFUL#{i+1}" if cfg.n_faithful > 1 else "FAITHFUL"
            user = mk_user(
                "FAITHFUL",
                "Translate as faithfully as possible. Prefer accuracy over elegance. Keep structure close when possible.",
                style_spec, glossary_payload
            )
            candidates[key] = self._translate_call(user, cfg.temp_faithful)

        for i in range(cfg.n_natural):
            key = f"NATURAL#{i+1}" if cfg.n_natural > 1 else "NATURAL"
            user = mk_user(
                "NATURAL",
                "Translate into natural, idiomatic Korean while preserving meaning exactly. Avoid awkward calques.",
                style_spec, glossary_payload
            )
            candidates[key] = self._translate_call(user, cfg.temp_natural)

        for i in range(cfg.n_reference):
            key = f"REFERENCE_LIKE#{i+1}" if cfg.n_reference > 1 else "REFERENCE_LIKE"
            user = mk_user(
                "REFERENCE-LIKE",
                "Translate into professional WMT-style written Korean for this domain. Keep terms consistent and punctuation clean.",
                style_spec, glossary_payload
            )
            candidates[key] = self._translate_call(user, cfg.temp_reference)

        return candidates

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
        adeq = self.llm.call_json(
            ADEQUACY_SYSTEM,
            adeq_user,
            temperature=0.0,
            max_tokens=self.llm.cfg.max_tokens_review,
            schema_hint="Return JSON with score_0_100 (int), critical (bool), issues (list).",
        )
        if not isinstance(adeq, dict):
            adeq = {}

        flu_user = self._render(
            FLUENCY_USER,
            STYLE_SPEC_JSON=style_spec,
            SOURCE=source,
            CANDIDATE_KO=candidate_ko,
        )
        flu = self.llm.call_json(
            FLUENCY_SYSTEM,
            flu_user,
            temperature=0.0,
            max_tokens=self.llm.cfg.max_tokens_review,
            schema_hint="Return JSON with score_0_100 (int), issues (list).",
        )
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
        fixlist = self.llm.call_json(
            SYNTH_SYSTEM,
            user,
            temperature=0.0,
            max_tokens=self.llm.cfg.max_tokens_synth,
            schema_hint="Return JSON with priority_fixes(list) and optional_polish(list).",
        )
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
        out = self.llm.call_text(
            EDITOR_SYSTEM,
            user,
            temperature=0.2,
            max_tokens=self.llm.cfg.max_tokens_translation,
            retries=3,
        )
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
        out = self.llm.call_text(
            EDITORLITE_SYSTEM,
            user,
            temperature=0.2,
            max_tokens=self.llm.cfg.max_tokens_translation,
            retries=3,
        )
        return clean_translation(out)

    def proofread(
        self,
        source: str,
        style_spec: Dict[str, Any],
        edited_ko: str,
    ) -> str:
        user = self._render(
            PROOF_USER,
            STYLE_SPEC_JSON=style_spec,
            SOURCE=source,
            EDITED_KO=edited_ko,
        )
        out = self.llm.call_text(
            PROOF_SYSTEM,
            user,
            temperature=0.0,
            max_tokens=self.llm.cfg.max_tokens_translation,
            retries=3,
        )
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
        gate = self.llm.call_json(
            GATE_SYSTEM,
            user,
            temperature=0.0,
            max_tokens=self.llm.cfg.max_tokens_gate,
            schema_hint="Return JSON with pass(bool) and violations(list).",
        )
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
                fixes.append({
                    "priority": pr,
                    "type": "FORMAT",
                    "instruction": instr,
                    "must_follow_glossary": True,
                })
                pr += 1
        return {"priority_fixes": fixes, "optional_polish": []}

    def select_candidate_without_review(self, source: str, candidates: Dict[str, str]) -> Tuple[str, str]:
        """
        Cheap heuristic selection when reviewers are disabled.
        Prefers candidates that preserve numbers/newlines; tie-breaker prefers REFERENCE_LIKE then FAITHFUL then NATURAL.
        """
        order_pref = {"REFERENCE_LIKE": 3, "REFERENCE_LIKE#1": 3, "FAITHFUL": 2, "NATURAL": 1}
        best_key = None
        best_score = -1e9
        best_text = ""
        for k, cand in candidates.items():
            score = 0.0
            ok_nums, missing = check_numbers_preserved(source, cand)
            score += 10.0 if ok_nums else -5.0 * len(missing)
            score += 5.0 if check_newlines_preserved(source, cand) else -5.0
            score += order_pref.get(k.split("#")[0], 0)
            if score > best_score:
                best_score = score
                best_key = k
                best_text = cand
        return best_key or list(candidates.keys())[0], best_text or list(candidates.values())[0]

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

        # 1) Choose base candidate
        if cfg.use_reviewers:
            best_key, best_text = None, ""
            best_score = -1e9
            best_adeq, best_flu = {}, {}

            for k, cand in candidates.items():
                adeq, flu, score = self.review_candidate(domain, source, style_spec, cand)
                if score > best_score:
                    best_score = score
                    best_key = k
                    best_text = cand
                    best_adeq, best_flu = adeq, flu

            chosen = best_text
            adeq_rep, flu_rep = best_adeq, best_flu
        else:
            best_key, chosen = self.select_candidate_without_review(source, candidates)
            adeq_rep, flu_rep = {}, {}

        # 2) Edit / refine
        out = chosen
        if cfg.use_editor:
            if cfg.use_reviewers:
                fixlist = self.synthesize_fixes(style_spec, glossary_payload, adeq_rep, flu_rep)
                out = self.edit_with_fixlist(domain, source, style_spec, glossary_payload, out, fixlist)
            else:
                out = self.edit_lite(domain, source, style_spec, glossary_payload, out)

        # 3) Proofread
        if cfg.use_proofreader:
            out = self.proofread(source, style_spec, out)

        # 4) Gatekeeper (+ optional repair loop)
        if cfg.use_gatekeeper:
            gate = self.gatekeep(source, style_spec, glossary_payload, out)
            if gate.get("pass") is False and cfg.use_repair_loop:
                fixlist2 = self.violations_to_fixlist(gate)
                out = self.edit_with_fixlist(domain, source, style_spec, glossary_payload, out, fixlist2)
                if cfg.use_proofreader:
                    out = self.proofread(source, style_spec, out)

        mem.add_context(source, out)
        return out


# =========================
# Evaluation & sampling
# =========================

def choose_reference(row: Dict[str, Any]) -> Optional[str]:
    # Prefer "target" if present; else "original_target"
    t = row.get("target", None)
    if isinstance(t, str) and t.strip():
        return t
    ot = row.get("original_target", None)
    if isinstance(ot, str) and ot.strip():
        return ot
    return None

def filter_rows(rows: List[Dict[str, Any]], exclude_bad_source: bool) -> List[int]:
    idxs = []
    for i, r in enumerate(rows):
        if exclude_bad_source and bool(r.get("is_bad_source", False)):
            continue
        ref = choose_reference(r)
        if ref is None:
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
    # Preserve original order
    sampled.sort()
    return sampled

def compute_metric(metric_name: str, hyps: List[str], refs: List[str]) -> float:
    if metric_name.lower() == "bleu":
        bleu = BLEU()
        score = bleu.corpus_score(hyps, [refs]).score
        return float(score)
    # default: chrF
    chrf = CHRF(word_order=2)  # chrF++ style
    score = chrf.corpus_score(hyps, [refs]).score
    return float(score)


# =========================
# Cache
# =========================

def subset_tag(sample_docs: int, seed: int, exclude_bad_source: bool) -> str:
    return f"docs{sample_docs}_seed{seed}_exclbad{int(exclude_bad_source)}"

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
            idx = int(obj["idx"])
            pred_map[idx] = obj["prediction"]
    return pred_map

def save_cached_predictions(run_dir: Path, pred_map: Dict[int, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for idx in sorted(pred_map.keys()):
            f.write(json.dumps({"idx": idx, "prediction": pred_map[idx]}, ensure_ascii=False) + "\n")


# =========================
# Running a config on a subset
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
    if cached is not None:
        # Ensure all subset present
        if all(i in cached for i in subset_idxs):
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

        # IMPORTANT: remove references before sending to model
        r.pop("target", None)
        r.pop("original_target", None)

        pred = pipe.translate_one(cfg, r)
        pred_map[idx] = pred

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

    hyps = []
    refs = []
    for idx in subset_idxs:
        hyp = pred_map[idx]
        ref = choose_reference(rows[idx]) or ""
        hyps.append(hyp)
        refs.append(ref)

    return compute_metric(metric, hyps, refs)


# =========================
# Stepwise ablation search
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
    max_steps: int = 999,
    features: Optional[List[str]] = None,
) -> Tuple[PipelineConfig, float, List[Dict[str, Any]]]:
    feats = features or list(ABLATION_FEATURES)

    history: List[Dict[str, Any]] = []

    cur_cfg = start_cfg.normalized()
    cur_score = evaluate_config(llm_cfg, cur_cfg, rows, subset_idxs, metric, cache_dir)

    history.append({
        "step": 0,
        "action": "start",
        "removed": "",
        "config_id": cur_cfg.id_str(),
        "score": cur_score,
        "config": cur_cfg.to_json(),
    })

    step = 0
    while step < max_steps:
        step += 1
        best_score = cur_score
        best_cfg = cur_cfg
        best_removed = ""

        # Try removing each enabled feature
        for feat in feats:
            if getattr(cur_cfg, feat, False) is False:
                continue
            cand_cfg = disable_feature(cur_cfg, feat)
            cand_score = evaluate_config(llm_cfg, cand_cfg, rows, subset_idxs, metric, cache_dir)
            history.append({
                "step": step,
                "action": "try_remove",
                "removed": feat,
                "config_id": cand_cfg.id_str(),
                "score": cand_score,
                "config": cand_cfg.to_json(),
            })
            if cand_score > best_score:
                best_score = cand_score
                best_cfg = cand_cfg
                best_removed = feat

        if best_cfg.id_str() != cur_cfg.id_str() and (best_score >= cur_score + min_improve):
            # Accept the removal that improves the score most
            cur_cfg = best_cfg
            cur_score = best_score
            history.append({
                "step": step,
                "action": "accept_remove",
                "removed": best_removed,
                "config_id": cur_cfg.id_str(),
                "score": cur_score,
                "config": cur_cfg.to_json(),
            })
        else:
            history.append({
                "step": step,
                "action": "stop",
                "removed": "",
                "config_id": cur_cfg.id_str(),
                "score": cur_score,
                "config": cur_cfg.to_json(),
            })
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

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Common args helper
    def add_common(p: argparse.ArgumentParser):
        p.add_argument("--input", required=True, help="Input JSONL (en-ko_KR.jsonl)")
        p.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
        p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt5-nano"))
        p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"))
        p.add_argument("--timeout", type=float, default=180.0)
        p.add_argument("--exclude_bad_source", action="store_true", help="Exclude is_bad_source rows for eval/run.")

    # run
    p_run = sub.add_parser("run", help="Run translation using a config and write output JSONL with predictions.")
    add_common(p_run)
    p_run.add_argument("--output", required=True)
    p_run.add_argument("--config", required=False, help="Path to config JSON. If omitted, uses full default config.")
    p_run.add_argument("--cache_dir", default="runs_cache", help="Cache directory (intermediate reuse).")

    # search
    p_search = sub.add_parser("search", help="Stepwise ablation search (remove one component at a time).")
    add_common(p_search)
    p_search.add_argument("--metric", choices=["chrf", "bleu"], default="chrf")
    p_search.add_argument("--sample_docs", type=int, default=50, help="Evaluate on N documents (0=all).")
    p_search.add_argument("--seed", type=int, default=13)
    p_search.add_argument("--cache_dir", default="runs_ablation")
    p_search.add_argument("--min_improve", type=float, default=0.0, help="Minimum improvement to accept a removal.")
    p_search.add_argument("--max_steps", type=int, default=50)

    args = ap.parse_args()

    rows = read_jsonl(args.input)

    llm_cfg = LLMConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    if args.cmd == "search":
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        idxs = filter_rows(rows, exclude_bad_source=args.exclude_bad_source)
        subset_idxs = sample_by_docs(rows, idxs, sample_docs=args.sample_docs, seed=args.seed)

        # Start from FULL pipeline config
        start_cfg = PipelineConfig().normalized()

        best_cfg, best_score, history = stepwise_backward_elimination(
            llm_cfg=llm_cfg,
            start_cfg=start_cfg,
            rows=rows,
            subset_idxs=subset_idxs,
            metric=args.metric,
            cache_dir=cache_dir,
            min_improve=args.min_improve,
            max_steps=args.max_steps,
            features=ABLATION_FEATURES,
        )

        # Save outputs
        (cache_dir / "best_config.json").write_text(json.dumps(best_cfg.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        write_history_csv(cache_dir / "search_results.csv", history)

        print("=== DONE ===")
        print(f"Metric: {args.metric}")
        print(f"Subset docs: {args.sample_docs}, rows: {len(subset_idxs)} (exclude_bad_source={args.exclude_bad_source})")
        print(f"Best score: {best_score:.4f}")
        print(f"Best config id: {best_cfg.id_str()}")
        print(f"Saved: {cache_dir / 'best_config.json'}")
        print(f"Saved: {cache_dir / 'search_results.csv'}")

    elif args.cmd == "run":
        cfg = PipelineConfig().normalized()
        if args.config:
            cfg_json = json.loads(Path(args.config).read_text(encoding="utf-8"))
            cfg = PipelineConfig.from_json(cfg_json).normalized()

        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        idxs = filter_rows(rows, exclude_bad_source=args.exclude_bad_source)
        # For run: use all eligible rows (or include all if you want; here we keep eligible set)
        subset_idxs = idxs

        pred_map = run_config_on_subset(
            llm_cfg=llm_cfg,
            cfg=cfg,
            rows=rows,
            subset_idxs=subset_idxs,
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
        print("=== DONE ===")
        print(f"Wrote: {args.output}")
        print(f"Config: {cfg.id_str()}")

    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
