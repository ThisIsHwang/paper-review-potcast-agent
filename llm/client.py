import json
import logging
import time
from typing import Any, Dict, Optional

from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: Optional[str], model: str):
        self.model = model
        # OpenAI client falls back to env var when api_key is None
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def generate_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logging.debug("LLM request attempt %d/%d", attempt, max_retries)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                logging.warning("LLM call failed (attempt %d): %s", attempt, exc)
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_error}")
