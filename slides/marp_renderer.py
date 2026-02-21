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
