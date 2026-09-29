"""Structural checks for dataset rows."""

from __future__ import annotations

from pathlib import Path

import yaml

from policylora.paths import repo_root


def taxonomy_labels(path: Path | None = None) -> set[str]:
    document = yaml.safe_load((path or repo_root() / "policylora/data/taxonomy.yaml").read_text())
    return {item["id"] for item in document["categories"]}


def automatic_checks(row: dict, labels: set[str] | None = None) -> list[str]:
    known = labels if labels is not None else taxonomy_labels()
    errors: list[str] = []
    if row["label"] not in known:
        errors.append("label")
    violating = row["label"] != "compliant" or row.get("tenant_violation") is True
    span = row.get("span")
    rewrite = row.get("compliant_rewrite")
    if violating:
        if not span or span not in row["message"]:
            errors.append("span")
        if not rewrite or (span and span in rewrite):
            errors.append("rewrite")
        if not row.get("rule") or not row.get("policy_match"):
            errors.append("citation")
    elif span:
        errors.append("compliant_span")
    return errors
