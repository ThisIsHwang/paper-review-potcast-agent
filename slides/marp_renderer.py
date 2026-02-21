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

import glob
import logging
import os
import subprocess
from typing import List


def render_markdown_to_images(md_path: str, output_prefix: str) -> List[str]:
    """
    Render markdown to PNG slides via marp.
    output_prefix should be a path prefix (e.g., /.../slides_2024-12-31_).
    We pass an explicit .png to marp so generated files include the extension.
    """
    output_path = f"{output_prefix}.png"
    cmd = [
        "marp",
        md_path,
        "--images",
        "png",
        "--output",
        output_path,
        "--allow-local-files",
    ]
    logging.info("Rendering slides via marp: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logging.error("marp CLI not found. Install @marp-team/marp-cli.")
        return []
    except subprocess.CalledProcessError as exc:
        logging.error("marp rendering failed: %s", exc)
        return []

    pattern = f"{output_prefix}*.png"
    images = sorted(glob.glob(pattern))
    logging.info("Rendered %d slide images", len(images))
    return images
