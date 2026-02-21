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
from typing import Optional

import requests

from .models import HFPaperEntry
from storage import paths


def download_pdf(paper: HFPaperEntry, date: str, base_dir: str) -> Optional[str]:
    if not paper.pdf_url:
        logging.error("Paper %s missing pdf_url", paper.paper_id)
        return None

    target_dir = paths.paper_dir(base_dir, date, paper.paper_id)
    paths.ensure_dir(target_dir)
    pdf_path = os.path.join(target_dir, "paper.pdf")

    if os.path.exists(pdf_path):
        logging.info("PDF already cached: %s", pdf_path)
        return pdf_path

    logging.info("Downloading PDF for %s from %s", paper.paper_id, paper.pdf_url)
    try:
        with requests.get(paper.pdf_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return pdf_path
    except Exception as exc:
        logging.error("Failed to download %s: %s", paper.paper_id, exc)
        return None
