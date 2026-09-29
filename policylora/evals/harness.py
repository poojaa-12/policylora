"""CPU eval harness. The rules baseline is measured here. Model baselines replay a cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from policylora.data.generate import FINGERPRINT_PATHS, GOLD_PATH, OOD_PATH, TENANT_PATH, fingerprint
from policylora.data.load import read_jsonl
from policylora.evals.metrics import category_scores, percentile, safe_div, span_correct
from policylora.evals.report import render_report
from policylora.paths import repo_root
from policylora.serve.rules import RulesEngine

ROOT = repo_root()
RESULTS_PATH = ROOT / "policylora/evals/results/latest.json"
REPORT_PATH = ROOT / "policylora/evals/results/report.md"
CACHE_DIR = ROOT / "policylora/evals/cache"
CACHE_FILES = {
    "untuned_base": "base.jsonl",
    "policylora": "policylora.jsonl",
    "frontier": "frontier.jsonl",
}


def _predict(rules: RulesEngine, row: dict) -> tuple[str, str | None, float]:
    started = time.perf_counter()
    outcome = rules.evaluate(row["message"])
    elapsed = (time.perf_counter() - started) * 1000
    if not outcome.violations or outcome.verdict == "pass":
        return "compliant", None, elapsed
    chosen = next((item for item in outcome.violations if item.category == row["label"]), outcome.violations[0])
    return chosen.category or "violation", chosen.span, elapsed


def score_labeled(rows: list[dict], rules: RulesEngine) -> dict:
    pairs = []
    latencies = []
    obvious_hits = 0
    obvious_total = 0
    span_hits = 0
    hard_fp = 0
    hard_total = 0
    for row in rows:
        predicted, span, elapsed = _predict(rules, row)
        pairs.append((row["label"], predicted))
        latencies.append(elapsed)
        if row.get("hard_negative"):
            hard_total += 1
            if predicted != "compliant":
                hard_fp += 1
        if row["subtlety"] == "obvious" and row["label"] != "compliant" and not row.get("hard_negative"):
            obvious_total += 1
            if predicted == row["label"]:
                obvious_hits += 1
            if span_correct(row.get("span"), span):
                span_hits += 1
    return {
        "per_category": category_scores(pairs),
        "obvious_violation_recall": safe_div(obvious_hits, obvious_total),
        "hard_negative_fpr": safe_div(hard_fp, hard_total),
        "span_accuracy": safe_div(span_hits, obvious_total),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "n": len(rows),
    }


def tenant_accuracy(rows: list[dict], rules: RulesEngine) -> float | None:
    if not rows:
        return None
    hits = 0
    for row in rows:
        predicted, _, _ = _predict(rules, row)
        predicted_violation = predicted != "compliant"
        if predicted_violation == bool(row.get("tenant_violation")):
            hits += 1
    return hits / len(rows)


def _cached_baseline(name: str, rows: list[dict]) -> dict:
    path = CACHE_DIR / CACHE_FILES[name]
    if not path.exists():
        return not_measured("not_measured")
    predictions = {}
    for row in read_jsonl(path):
        predictions[row["id"]] = row
    return score_predictions(rows, predictions)


def not_measured(reason: str) -> dict:
    return {"status": reason}


def score_predictions(rows: list[dict], predictions: dict[str, dict]) -> dict:
    pairs = []
    latencies = []
    hard_fp = 0
    hard_total = 0
    span_hits = 0
    span_total = 0
    missing = 0
    for row in rows:
        predicted = predictions.get(row["id"])
        if predicted is None:
            missing += 1
            continue
        pairs.append((row["label"], predicted["label"]))
        if "latency_ms" in predicted:
            latencies.append(predicted["latency_ms"])
        if row.get("hard_negative"):
            hard_total += 1
            if predicted["label"] != "compliant":
                hard_fp += 1
        if row["label"] != "compliant" and row.get("span"):
            span_total += 1
            if span_correct(row.get("span"), predicted.get("span")):
                span_hits += 1
    categories = category_scores(pairs)
    recalls = [item["recall"] for item in categories.values() if item["recall"] is not None]
    return {
        "status": "measured",
        "coverage": (len(rows) - missing) / len(rows) if rows else None,
        "per_category": categories,
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "hard_negative_fpr": safe_div(hard_fp, hard_total),
        "span_accuracy": safe_div(span_hits, span_total),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "n": len(rows) - missing,
    }


def run() -> dict:
    started = time.perf_counter()
    rules = RulesEngine()
    gold = read_jsonl(GOLD_PATH)
    ood = read_jsonl(OOD_PATH)
    tenant = read_jsonl(TENANT_PATH)
    rules_score = score_labeled(gold, rules)
    rules_score["tenant_policy_accuracy"] = tenant_accuracy(tenant, rules)
    rules_score["ood"] = score_labeled(ood, rules)
    results = {
        "fingerprint": fingerprint(FINGERPRINT_PATHS),
        "suite_wall_clock_s": round(time.perf_counter() - started, 3),
        "counts": {"gold": len(gold), "ood": len(ood), "tenant_eval": len(tenant)},
        "baselines": {
            "rules": rules_score,
            "untuned_base": _cached_baseline("untuned_base", gold),
            "policylora": _cached_baseline("policylora", gold),
            "frontier": _cached_baseline("frontier", gold),
        },
        "cost": {
            "status": "not_measured",
            "method": "slm hourly rental divided by measured throughput; frontier token price times measured tokens",
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    REPORT_PATH.write_text(render_report(results))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PolicyLoRA eval harness")
    parser.parse_args(argv)
    results = run()
    rules = results["baselines"]["rules"]
    print(
        json.dumps(
            {
                "obvious_violation_recall": rules["obvious_violation_recall"],
                "hard_negative_fpr": rules["hard_negative_fpr"],
                "tenant_policy_accuracy": rules["tenant_policy_accuracy"],
                "wall_clock_s": results["suite_wall_clock_s"],
            }
        )
    )


if __name__ == "__main__":
    main()
