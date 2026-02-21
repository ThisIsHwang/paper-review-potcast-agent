#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Naive few-shot prompt baseline for WMT24++ en->ko_KR
- LangChain ChatOpenAI + OpenAI gpt-4o-mini
- Prompting only: system + few-shot examples + one-shot translate
- NO agents, NO reviewers, NO post-editing, NO training
- IMPORTANT: Never leak references (target/original_target) into prompts.
- Optional offline eval with sacrebleu (BLEU/chrF) using target/original_target.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sacrebleu.metrics import BLEU, CHRF


# -----------------------
# IO
# -----------------------

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


# -----------------------
# LLM (ChatOpenAI)
# -----------------------

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout: float = 180.0

    # For deterministic baseline
    temperature: float = 0.0

    # Output cap (Chat Completions: max_tokens is still supported for non o-series)
    max_tokens: int = 1200


class LLMCaller:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        # LangChain 버전별 파라미터 차이를 흡수
        try:
            self.llm = ChatOpenAI(
                model=cfg.model,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=cfg.timeout,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        except TypeError:
            # older langchain-openai
            self.llm = ChatOpenAI(
                model=cfg.model,
                openai_api_key=cfg.api_key,
                openai_api_base=cfg.base_url,
                request_timeout=cfg.timeout,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

    def call(self, system: str, user: str, retries: int = 3) -> str:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.llm.invoke(msgs)
                return (resp.content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        raise RuntimeError(f"LLM call failed: {last_err}")


# -----------------------
# Prompt (naive few-shot)
# -----------------------

SYSTEM_PROMPT = """\
You are a professional English→Korean translator (ko-KR).
Rules:
- Output ONLY the Korean translation text. No explanations, no labels.
- Preserve meaning exactly. Do not add or omit information.
- Preserve all numbers, dates, names, symbols, and line breaks exactly.
- Keep URLs, emails, @handles, and #hashtags unchanged.
- Use natural, standard written Korean (neutral formal: "~다" style), unless the source is clearly casual social text.
"""

# 데이터 유출 없는 범용 few-shot 예시
FEWSHOT_EXAMPLES: List[Tuple[str, str]] = [
    (
        "The company reported revenue of $3.2 billion in 2024.",
        "해당 회사는 2024년에 32억 달러의 매출을 보고했다."
    ),
    (
        "“We will not raise taxes,” the minister said.",
        "“우리는 세금을 인상하지 않을 것”이라고 장관은 말했다."
    ),
    (
        "Follow @OpenAI and visit https://openai.com for updates.",
        "@OpenAI를 팔로우하고 업데이트를 보려면 https://openai.com 을 방문하라."
    ),
    (
        "The meeting will be held in Paris (France) on May 12, 2025.",
        "회의는 2025년 5월 12일에 파리(프랑스)에서 열릴 예정이다."
    ),
]

def build_user_prompt(domain: str, source: str, fewshots: List[Tuple[str, str]]) -> str:
    lines = []
    lines.append("Translate English to Korean (ko-KR).")
    lines.append("Return Korean translation only.\n")
    lines.append(f"[domain] {domain}\n")
    lines.append("Examples:")
    for en, ko in fewshots:
        lines.append(f"English: {en}")
        lines.append(f"Korean: {ko}")
        lines.append("")
    lines.append("Now translate:")
    lines.append(f"English: {source}")
    lines.append("Korean:")
    return "\n".join(lines)


# -----------------------
# Postprocess
# -----------------------

def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:text)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

def postprocess_translation(text: str) -> str:
    t = strip_code_fences(text)

    # 혹시 라벨을 붙이면 제거
    for prefix in ["Korean:", "한국어:", "번역:", "번역문:", "KO:"]:
        if t.startswith(prefix):
            t = t[len(prefix):].lstrip()

    # 전체가 따옴표로 감싸진 경우 제거
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()

    return t


# -----------------------
# Filtering + eval
# -----------------------

def choose_reference(row: Dict[str, Any]) -> Optional[str]:
    t = row.get("target")
    if isinstance(t, str) and t.strip():
        return t
    ot = row.get("original_target")
    if isinstance(ot, str) and ot.strip():
        return ot
    return None

def compute_metric(metric: str, hyps: List[str], refs: List[str]) -> float:
    if metric.lower() == "bleu":
        # ✅ Korean BLEU 토크나이저 적용
        m = BLEU(tokenize="ko-mecab")
        return float(m.corpus_score(hyps, [refs]).score)

    # chrF는 문자 n-gram 기반이라 토크나이저 영향이 거의 없어서 그대로 두어도 됩니다.
    m = CHRF(word_order=2)  # chrF++
    return float(m.corpus_score(hyps, [refs]).score)

# -----------------------
# Commands
# -----------------------

def cmd_run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    # 100 test
    rows = rows[:100]
    llm_cfg = LLMConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    caller = LLMCaller(llm_cfg)

    out_rows: List[Dict[str, Any]] = []
    for r in tqdm(rows, desc="Naive few-shot translating (gpt-4o-mini)"):
        rr = dict(r)

        domain = str(rr.get("domain", "") or "unknown")
        source = str(rr.get("source", "") or rr.get("src", "") or "")

        # Keep references for offline eval, but NEVER send them to the model
        ref_target = rr.get("target")
        ref_orig_target = rr.get("original_target")

        # IMPORTANT: never leak references into prompts
        rr.pop("target", None)
        rr.pop("original_target", None)

        user = build_user_prompt(domain, source, FEWSHOT_EXAMPLES)
        raw = caller.call(SYSTEM_PROMPT, user)
        pred = postprocess_translation(raw)

        # Restore refs for downstream eval
        if ref_target is not None:
            rr["target"] = ref_target
        if ref_orig_target is not None:
            rr["original_target"] = ref_orig_target
        rr["prediction"] = pred
        out_rows.append(rr)

    write_jsonl(args.output, out_rows)
    print(f"Done. Wrote: {args.output}")

def cmd_eval(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)

    ref_rows: List[Dict[str, Any]] = []
    ref_by_key: Dict[Tuple[Any, Any], Optional[str]] = {}
    if getattr(args, "ref", None):
        ref_rows = read_jsonl(args.ref)
        for rr in ref_rows:
            ref = choose_reference(rr)
            key = (rr.get("document_id"), rr.get("segment_id"))
            if ref is not None and key not in ref_by_key:
                ref_by_key[key] = ref

    hyps, refs = [], []
    skipped = 0
    for idx, r in enumerate(rows):
        if args.exclude_bad_source and bool(r.get("is_bad_source", False)):
            continue
        hyp = r.get("prediction")
        if not isinstance(hyp, str):
            hyp = ""

        # Try to locate a reference in the prediction row, or fall back to a ref file
        ref = choose_reference(r)
        if (not isinstance(ref, str) or not ref.strip()) and ref_by_key:
            key = (r.get("document_id"), r.get("segment_id"))
            ref = ref_by_key.get(key)
            if (not isinstance(ref, str) or not ref.strip()) and idx < len(ref_rows):
                ref = choose_reference(ref_rows[idx])

        if not isinstance(ref, str) or not ref.strip():
            skipped += 1
            continue

        hyps.append(hyp)
        refs.append(ref)

    if not hyps:
        raise SystemExit("No hypotheses to score (prediction/target missing after filtering).")

    score = compute_metric(args.metric, hyps, refs)
    print(f"{args.metric.upper()} = {score:.4f}")
    if skipped:
        print(f"(skipped rows without reference: {skipped})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--output", required=True)

    p_run.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    p_run.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p_run.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    p_run.add_argument("--timeout", type=float, default=180.0)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--max_tokens", type=int, default=1200)

    # eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--input", required=True)
    p_eval.add_argument("--metric", choices=["chrf", "bleu"], default="chrf")
    p_eval.add_argument("--exclude_bad_source", action="store_true")
    p_eval.add_argument("--ref", required=False, help="Optional reference JSONL (original data) if predictions file lacks targets.")

    args = ap.parse_args()

    if args.cmd == "run":
        if not args.api_key:
            raise SystemExit("OPENAI_API_KEY is missing. Set env var or pass --api_key.")
        cmd_run(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
