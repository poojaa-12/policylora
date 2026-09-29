"""Write gold, OOD, tenant-eval, and the synthetic split files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from policylora.data.catalog import assign_train_splits, build_gold, build_ood, build_tenant_eval, build_train_candidates
from policylora.data.checks import automatic_checks, taxonomy_labels
from policylora.data.dedup import build_index, dedupe
from policylora.data.load import LeakageError, assert_trainable, read_jsonl, write_jsonl
from policylora.paths import repo_root
from policylora.serve.rules import RulesEngine

ROOT = repo_root()
GOLD_PATH = ROOT / "policylora/data/gold/gold.jsonl"
OOD_PATH = ROOT / "policylora/data/ood/ood.jsonl"
TENANT_PATH = ROOT / "policylora/data/gold/tenant_eval.jsonl"
GENERATED = ROOT / "policylora/data/generated"
MANIFEST = ROOT / "policylora/data/MANIFEST.sha256"
HOLDOUT_PATHS = [GOLD_PATH, OOD_PATH, TENANT_PATH]
FINGERPRINT_PATHS = HOLDOUT_PATHS + [
    ROOT / "policylora/data/taxonomy.yaml",
    ROOT / "policylora/data/restricted_products.yaml",
    ROOT / "policylora/serve/adapters/registry.json",
]


def fingerprint(paths: list[Path] | None = None) -> str:
    digest = hashlib.sha256()
    for path in paths or FINGERPRINT_PATHS:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_splits(train_size: int = 6000) -> dict[str, list[dict]]:
    holdouts = build_gold() + build_ood() + build_tenant_eval()
    holdout_pairs = [(row["label"], row["message"]) for row in holdouts]
    candidates = assign_train_splits(build_train_candidates())
    labels = taxonomy_labels()
    kept_structural = []
    review = []
    for row in candidates:
        errors = automatic_checks(row, labels)
        if errors:
            review.append({**row, "reasons": errors})
            continue
        kept_structural.append(row)
    deduped = dedupe(kept_structural, holdout_pairs, threshold=0.9, index=build_index())
    # Cap after dedup, preserving reserved-cell test rows.
    reserved_rows = [row for row in deduped if row["cell"].endswith("promissory_claim|social|retail|etf|ai|subtle|clean|base") or _is_reserved(row)]
    others = [row for row in deduped if row not in reserved_rows]
    trimmed = reserved_rows + others[: max(0, train_size - len(reserved_rows))]
    splits = {"train": [], "val": [], "test": [], "review": review}
    for row in trimmed:
        row["id"] = f"syn-{row['split']}-{len(splits[row['split']]):05d}"
        splits[row["split"]].append(row)
    return splits


def _is_reserved(row: dict) -> bool:
    return row.get("cell") == "promissory_claim|social|retail|etf|ai|subtle|clean|base"


def write_holdouts() -> dict[str, int]:
    gold = build_gold()
    ood = build_ood()
    tenant = build_tenant_eval()
    write_jsonl(GOLD_PATH, gold)
    write_jsonl(OOD_PATH, ood)
    write_jsonl(TENANT_PATH, tenant)
    _write_manifest()
    return {"gold": len(gold), "ood": len(ood), "tenant_eval": len(tenant)}


def write_generated(train_size: int) -> dict[str, int]:
    splits = build_splits(train_size)
    counts = {}
    for name, rows in splits.items():
        write_jsonl(GENERATED / f"{name}.jsonl", rows)
        counts[name] = len(rows)
    blocked = {row["id"] for path in HOLDOUT_PATHS for row in read_jsonl(path)}
    assert_trainable(splits["train"], blocked)
    return counts


def _write_manifest() -> None:
    lines = []
    for path in HOLDOUT_PATHS:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    MANIFEST.write_text("\n".join(lines) + "\n")


def check() -> None:
    labels = taxonomy_labels()
    for path in HOLDOUT_PATHS:
        if not path.exists():
            raise SystemExit(f"missing {path}")
        for row in read_jsonl(path):
            errors = automatic_checks(row, labels)
            if errors:
                raise SystemExit(f"{path} {row['id']} {errors}")
    ids: list[str] = []
    for path in HOLDOUT_PATHS:
        ids.extend(row["id"] for row in read_jsonl(path))
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate holdout ids")
    recorded = MANIFEST.read_text().strip().splitlines()
    if len(recorded) != len(HOLDOUT_PATHS):
        raise SystemExit("manifest does not list every holdout")
    for line, path in zip(recorded, HOLDOUT_PATHS):
        digest, name = line.split("  ", 1)
        if name != path.relative_to(ROOT).as_posix():
            raise SystemExit(f"manifest path mismatch for {path}")
        if digest != hashlib.sha256(path.read_bytes()).hexdigest():
            raise SystemExit(f"manifest hash mismatch for {path}")
    train_path = GENERATED / "train.jsonl"
    if train_path.exists():
        blocked = set(ids)
        try:
            assert_trainable(read_jsonl(train_path), blocked)
        except LeakageError as exc:
            raise SystemExit(str(exc)) from exc


def route_disagreement(row: dict, rules: RulesEngine) -> str:
    if row.get("reviewed"):
        return "keep"
    outcome = rules.evaluate(row["message"])
    predicted = "compliant" if outcome.verdict == "pass" else (outcome.violations[0].category or "violation")
    if predicted != row["label"]:
        return "review"
    return "keep"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build PolicyLoRA datasets")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--holdout-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check()
        print(json.dumps({"status": "ok", "fingerprint": fingerprint()}))
        return
    counts = write_holdouts()
    if not args.holdout_only:
        counts.update(write_generated(args.train_size))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
