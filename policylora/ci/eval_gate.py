"""CI gate. Recomputes nothing from a GPU; it checks the committed eval artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from policylora.data.generate import fingerprint
from policylora.paths import repo_root

ROOT = repo_root()
DEFAULT_RESULTS = ROOT / "policylora/evals/results/latest.json"
DEFAULT_THRESHOLDS = ROOT / "policylora/evals/thresholds.yaml"


def gate(results: dict, thresholds: dict, current_fingerprint: str | None = None) -> list[str]:
    errors: list[str] = []
    expected = current_fingerprint if current_fingerprint is not None else fingerprint()
    if results.get("fingerprint") != expected:
        errors.append("eval artifact is stale for the current data, taxonomy, or adapter registry")
    rules = thresholds["rules"]
    if rules.get("required", False):
        measured = results["baselines"]["rules"]
        recall = measured["obvious_violation_recall"]
        fpr = measured["hard_negative_fpr"]
        if recall is None or recall < rules["min_obvious_violation_recall"]:
            errors.append(f"obvious violation recall {recall} is below {rules['min_obvious_violation_recall']}")
        if fpr is None or fpr > rules["max_hard_negative_fpr"]:
            errors.append(f"hard-negative false-positive rate {fpr} is above {rules['max_hard_negative_fpr']}")
    if thresholds["slm"].get("required", False):
        slot = results["baselines"]["policylora"]
        if slot.get("status") == "not_measured" or "macro_recall" not in slot:
            errors.append("adapter or data changed and the PolicyLoRA prediction cache was not refreshed")
        else:
            floor = thresholds["slm"].get("min_macro_recall")
            if floor is not None and slot["macro_recall"] < floor:
                errors.append(f"PolicyLoRA macro recall {slot['macro_recall']} is below {floor}")
            ceiling = thresholds["slm"].get("max_hard_negative_fpr")
            if ceiling is not None and slot["hard_negative_fpr"] > ceiling:
                errors.append(f"PolicyLoRA hard-negative FPR {slot['hard_negative_fpr']} is above {ceiling}")
    return errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fail the build when the eval artifact is stale or below threshold")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args(argv)
    if not args.results.exists():
        print("missing eval artifact", file=sys.stderr)
        raise SystemExit(1)
    results = json.loads(args.results.read_text())
    thresholds = yaml.safe_load(args.thresholds.read_text())
    errors = gate(results, thresholds)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"status": "ok", "fingerprint": results["fingerprint"]}))


if __name__ == "__main__":
    main()
