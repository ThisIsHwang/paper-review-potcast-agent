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

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HFPaperEntry:
    paper_id: str
    title: str
    summary: str
    authors: List[str]
    upvotes: int
    published_at: str
    hf_url: Optional[str] = None
    arxiv_url: Optional[str] = None
    pdf_url: Optional[str] = None
    origin: Optional[str] = None


@dataclass
class ExtractedText:
    abstract: str = ""
    intro: str = ""
    conclusion: str = ""
    full_text: str = ""
