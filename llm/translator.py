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

from typing import List, Tuple

from .client import LLMClient


TRANSLATION_SYSTEM_PROMPT = (
    "You are a precise bilingual translator for spoken video narration. "
    "Preserve meaning, pacing, and technical terms while making the output natural to speak. "
    "When the target language is Korean, use a lively YouTuber host tone without changing facts."
)

LANGUAGE_NAMES = {
    "en": "English",
    "ko": "Korean",
}


def language_display(lang_code: str) -> str:
    """Return a human-friendly language label."""
    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code)


def _tone_guidance(target_language: str) -> str:
    if target_language.lower() != "ko":
        return ""
    return """
Korean tone requirements:
- Lively YouTuber host vibe: upbeat, friendly, energetic.
- Natural spoken Korean, not stiff translation; short sentences and smooth rhythm.
- Preserve all facts, numbers, and technical terms exactly; do not add or omit content.
"""


def translate_scripts(scripts: List[str], llm_client: LLMClient, target_language: str) -> List[str]:
    """
    Translate slide narration scripts into the requested language while keeping order and count.
    """
    if not scripts:
        return []

    language_name = language_display(target_language)
    numbered = "\n".join(f"{idx}. {text}" for idx, text in enumerate(scripts, 1))
    tone_guidance = _tone_guidance(target_language)
    prompt = f"""Translate the following slide narration scripts into {language_name} for voice-over.
Keep the tone conversational, mirror technical terms, and preserve the list order and length.
Do NOT add leading numbering (e.g., "1." or "1)") in the translated text.
{tone_guidance}
Return JSON with a single key "translations" as an array of strings matching the original count.

Original scripts (numbered):
{numbered}"""

    raw = llm_client.generate_json(TRANSLATION_SYSTEM_PROMPT, prompt)
    translations = raw.get("translations")
    if not isinstance(translations, list):
        raise ValueError("Translation response missing 'translations' list")

    cleaned = [str(item).strip() for item in translations]
    if len(cleaned) != len(scripts):
        raise ValueError(f"Translation length mismatch: expected {len(scripts)}, got {len(cleaned)}")

    return cleaned


def translate_scripts_and_instructions(
    scripts: List[str],
    instructions: List[str],
    llm_client: LLMClient,
    target_language: str,
) -> Tuple[List[str], List[str]]:
    """Translate scripts and their paired instructions together to keep them aligned."""
    if len(instructions) != len(scripts):
        raise ValueError("scripts and instructions length mismatch")

    language_name = language_display(target_language)
    numbered_scripts = "\n".join(f"{idx}. {text}" for idx, text in enumerate(scripts, 1))
    numbered_instr = "\n".join(f"{idx}. {text}" for idx, text in enumerate(instructions, 1))

    tone_guidance = _tone_guidance(target_language)
    prompt = f"""
Your task is to translate the slide narrations and their delivery instructions into {language_name}.

**Overall goals**
- Produce translations that sound fully native, natural, and engaging for a podcast listener.
- Avoid literal or stiff phrasing—rewrite as needed while preserving the speaker's intent, emotional tone, and pacing.
- Maintain the original meaning, nuance, and order of items.
- Keep the counts and alignment between scripts and instructions exactly the same.
- Do NOT add any new numbering (e.g., "1." or "1)").
{tone_guidance}
- The translated version should never feel like a translation; it should feel like it was originally written in {language_name}.
- Make the flow lively and dynamic so the audience never feels bored.

**Style guidelines**
- Use conversational, listener-friendly language.
- Improve clarity, rhythm, and narrative energy while staying faithful to the source.
- When the original text is dry, you may slightly enhance mood, tone, or transitions to sound more like a spoken podcast—without adding new factual content.
- Delivery instructions should guide vocal tone, pacing, emphasis, or mood naturally in {language_name}.

Return JSON ONLY:
{{
  "scripts": ["..."],
  "instructions": ["..."]
}}

Original scripts (numbered):
{numbered_scripts}

Original instructions (numbered, may be empty strings):
{numbered_instr}
"""

    raw = llm_client.generate_json(TRANSLATION_SYSTEM_PROMPT, prompt)
    scripts_out = raw.get("scripts")
    instructions_out = raw.get("instructions")
    if not isinstance(scripts_out, list) or not isinstance(instructions_out, list):
        raise ValueError("Translation response missing 'scripts' or 'instructions' lists")

    scripts_clean = [str(item).strip() for item in scripts_out]
    instr_clean = [str(item).strip() for item in instructions_out]

    if len(scripts_clean) != len(scripts) or len(instr_clean) != len(instructions):
        raise ValueError("Translation length mismatch between input and output")

    return scripts_clean, instr_clean
