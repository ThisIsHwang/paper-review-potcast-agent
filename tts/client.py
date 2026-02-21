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
from typing import List, Optional

from openai import OpenAI


def _strip_delivery_cues(text: str) -> str:
    cleaned = re.sub(r"\\[[^\\[\\]]+?\\]", "", text)
    cleaned = re.sub(r"\\s{2,}", " ", cleaned)
    return cleaned.strip()


class TTSClient:
    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        voice: str,
        style_instruction: Optional[str] = None,
        speed: float = 1.0,
    ):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.voice = voice
        self.style_instruction = (style_instruction or "").strip()
        self.speed = max(0.5, min(speed, 4.0))

    def synthesize(self, text: str, out_path: str, instruction: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        logging.debug("Synthesizing audio to %s", out_path)
        prompt = re.sub(r"^\\s*\\d+\\s*[.\\)-:]?\\s*", "", text.strip())
        prompt = _strip_delivery_cues(prompt)
        # Do not prepend instructions to the spoken text to avoid them being read aloud.
        resp = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=prompt,
            speed=self.speed,
        )
        with open(out_path, "wb") as f:
            f.write(resp.read())

    def synthesize_scripts(
        self,
        scripts: List[str],
        base_dir: str,
    ) -> List[str]:
        os.makedirs(base_dir, exist_ok=True)
        audio_files: List[str] = []
        for idx, script in enumerate(scripts, 1):
            out_path = os.path.join(base_dir, f"audio_slide_{idx:03d}.mp3")
            try:
                self.synthesize(script, out_path)
                audio_files.append(out_path)
            except Exception as exc:
                logging.error("TTS failed for slide %d: %s", idx, exc)
        return audio_files
