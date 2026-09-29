"""Scenario-cell splits. A whole cell lands in one split."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from policylora.paths import repo_root


def cell_key(row: dict) -> str:
    author = row["author_type"]
    author_value = author if isinstance(author, str) else author.value
    return "|".join(
        [
            row["label"],
            row["channel"],
            row["audience"],
            row["product"],
            author_value,
            row["subtlety"],
            row["noise"],
            row.get("tenant_id") or "base",
        ]
    )


def load_scenario(path: Path | None = None) -> dict:
    return yaml.safe_load((path or repo_root() / "policylora/data/scenario.yaml").read_text())


def reserved_keys(scenario: dict | None = None) -> set[str]:
    document = scenario if scenario is not None else load_scenario()
    keys = set()
    for cell in document["reserved_test_cells"]:
        keys.add(
            "|".join(
                [
                    cell["label"],
                    cell["channel"],
                    cell["audience"],
                    cell["product"],
                    cell["author_type"],
                    cell["subtlety"],
                    cell["noise"],
                    cell["tenant"],
                ]
            )
        )
    return keys


def split_for_cell(key: str, reserved: set[str]) -> str:
    if key in reserved:
        return "test"
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def assign_splits(rows: list[dict], reserved: set[str] | None = None) -> list[dict]:
    held = reserved if reserved is not None else reserved_keys()
    for row in rows:
        key = cell_key(row)
        row["cell"] = key
        row["split"] = split_for_cell(key, held)
    return rows
