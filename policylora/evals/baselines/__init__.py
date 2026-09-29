"""Frontier baseline. Live calls happen only with an API key; otherwise the cache is replayed."""

from __future__ import annotations

import json
import os
from typing import Callable

from policylora.paths import input_hash


def lookup(cache: dict[str, dict], model_id: str, body: str) -> dict | None:
    return cache.get(f"{model_id}:{input_hash(body)}")


def frontier_predict(
    body: str,
    *,
    cache: dict[str, dict],
    model_id: str,
    client: Callable[[str], dict] | None = None,
    api_key: str | None = None,
) -> dict | None:
    cached = lookup(cache, model_id, body)
    if cached is not None:
        return cached
    key = api_key if api_key is not None else os.environ.get("POLICLORA_FRONTIER_API_KEY")
    if not key or client is None:
        return None
    result = client(body)
    result["input_hash"] = input_hash(body)
    result["model_id"] = model_id
    return result


def load_cache(text: str) -> dict[str, dict]:
    found = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[f"{row['model_id']}:{row['input_hash']}"] = row
    return found
