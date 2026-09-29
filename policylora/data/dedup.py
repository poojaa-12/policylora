"""Near-duplicate removal. Jaccard by default; embeddings when configured."""

from __future__ import annotations

import math
import os
import re


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9%']+", text.lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class JaccardIndex:
    def __init__(self) -> None:
        self._items: dict[str, list[set[str]]] = {}

    def add(self, label: str, text: str) -> None:
        self._items.setdefault(label, []).append(tokens(text))

    def too_close(self, label: str, text: str, threshold: float) -> bool:
        current = tokens(text)
        return any(jaccard(current, previous) >= threshold for previous in self._items.get(label, []))


class VectorIndex:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self._items: dict[str, list[list[float]]] = {}

    def add(self, label: str, text: str) -> None:
        self._items.setdefault(label, []).append(self.embedder(text))

    def too_close(self, label: str, text: str, threshold: float) -> bool:
        current = self.embedder(text)
        return any(cosine(current, previous) >= threshold for previous in self._items.get(label, []))


def dedupe(rows: list[dict], holdout_texts: list[tuple[str, str]], threshold: float = 0.9, index=None) -> list[dict]:
    chosen = index or JaccardIndex()
    for label, text in holdout_texts:
        chosen.add(label, text)
    kept: list[dict] = []
    for row in rows:
        label = row["label"]
        if chosen.too_close(label, row["message"], threshold):
            continue
        chosen.add(label, row["message"])
        kept.append(row)
    return kept


def build_index():
    if os.environ.get("POLICLORA_DEDUP") == "embeddings":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(text: str) -> list[float]:
            vector = model.encode([text], normalize_embeddings=True)[0]
            return [float(value) for value in vector]

        return VectorIndex(embed)
    return JaccardIndex()
