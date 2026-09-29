"""Public rule excerpts and optional AWC PDF extraction."""

from __future__ import annotations

import re
from pathlib import Path


def parse_excerpts(text: str) -> list[dict]:
    chunks = re.split(r"\n(?=## )", text.strip())
    records = []
    for chunk in chunks:
        if not chunk.startswith("## "):
            continue
        lines = chunk.splitlines()
        title = lines[0][3:].strip()
        url = ""
        quote_lines: list[str] = []
        mode = None
        for line in lines[1:]:
            if line.startswith("URL:"):
                url = line.split(":", 1)[1].strip()
                mode = None
            elif line.startswith("Quote:"):
                quote_lines = [line.split(":", 1)[1].strip()]
                mode = "quote"
            elif mode == "quote":
                quote_lines.append(line.strip())
        records.append({"rule": title, "url": url, "quote": " ".join(part for part in quote_lines if part)})
    return records


def extract_pdf_text(path: Path) -> str:
    import fitz

    document = fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in document)
    finally:
        document.close()


def load_awc_urls(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
