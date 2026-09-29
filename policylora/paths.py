"""Paths, hashing, and small shared helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parents[2]


def input_hash(body: str) -> str:
    return hashlib.sha256(body.strip().encode()).hexdigest()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def negated(text: str, start: int) -> bool:
    window = text[max(0, start - 48) : start]
    return re.search(r"\b(not|no|never|isn't|aren't)\b", window, flags=re.I) is not None
