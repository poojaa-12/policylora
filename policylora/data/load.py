"""Train-loader guards against gold, OOD, and non-train splits."""

from __future__ import annotations

import json
from pathlib import Path


class LeakageError(Exception):
    pass


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    path.write_text(payload + ("\n" if rows else ""))


def holdout_ids(paths: list[Path]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            found.add(row["id"])
    return found


def assert_trainable(rows: list[dict], blocked_ids: set[str]) -> None:
    leaked = [row["id"] for row in rows if row["id"] in blocked_ids or row.get("split_role") in {"gold", "ood", "tenant_eval"}]
    if leaked:
        raise LeakageError(f"holdout rows in train: {leaked[:5]}")
    wrong_split = [row["id"] for row in rows if row.get("split") != "train"]
    if wrong_split:
        raise LeakageError(f"non-train split in train file: {wrong_split[:5]}")
