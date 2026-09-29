from __future__ import annotations

import json

import yaml

from policylora.ci.eval_gate import gate
from policylora.evals.baselines import frontier_predict, load_cache
from policylora.evals.metrics import frontier_cost_per_1k, percentile, slm_cost_per_1k
from policylora.paths import input_hash
from policylora.train.sft import ablation_configs, load_config, render_example, select_rows


def test_cost_and_percentiles():
    assert slm_cost_per_1k(2.0, 1000) == 2.0
    assert abs(frontier_cost_per_1k(1000, 100, 1.0, 2.0) - 1.2) < 1e-9
    assert percentile([1, 2, 3, 4], 0.95) > 3


def test_frontier_replays_cache_without_a_key():
    body = "hello"
    model = "frontier"
    cache = load_cache(
        json.dumps({"model_id": model, "input_hash": input_hash(body), "label": "compliant"}) + "\n"
    )
    found = frontier_predict(body, cache=cache, model_id=model, api_key=None, client=None)
    assert found["label"] == "compliant"
    assert frontier_predict("missing", cache=cache, model_id=model, api_key=None) is None


def test_gate_rejects_a_stale_artifact_and_a_missing_slm_cache():
    results = {
        "fingerprint": "old",
        "baselines": {
            "rules": {"obvious_violation_recall": 1.0, "hard_negative_fpr": 0.0},
            "policylora": {"status": "not_measured"},
        },
    }
    thresholds = {
        "rules": {"required": True, "min_obvious_violation_recall": 0.95, "max_hard_negative_fpr": 0.02},
        "slm": {"required": False},
    }
    assert gate(results, thresholds, current_fingerprint="new")
    thresholds["slm"]["required"] = True
    results["fingerprint"] = "new"
    errors = gate(results, thresholds, current_fingerprint="new")
    assert any("cache" in error for error in errors)


def test_prediction_cache_scores_and_committed_gate_passes():
    from policylora.ci.eval_gate import DEFAULT_RESULTS, DEFAULT_THRESHOLDS, gate
    from policylora.evals.harness import score_predictions

    scored = score_predictions(
        [
            {"id": "a", "label": "promissory_claim", "span": "guaranteed", "subtlety": "obvious", "hard_negative": False},
            {"id": "b", "label": "compliant", "span": None, "subtlety": "obvious", "hard_negative": True},
        ],
        {
            "a": {"label": "promissory_claim", "span": "guaranteed", "latency_ms": 12},
            "b": {"label": "compliant", "span": None, "latency_ms": 9},
        },
    )
    assert scored["status"] == "measured"
    assert scored["macro_recall"] == 1
    assert scored["hard_negative_fpr"] == 0
    results = json.loads(DEFAULT_RESULTS.read_text())
    thresholds = yaml.safe_load(DEFAULT_THRESHOLDS.read_text())
    assert gate(results, thresholds) == []


def test_training_format_and_ablation_grid():
    row = {
        "message": "The fund is guaranteed to return 8% a year.",
        "label": "promissory_claim",
        "rule": "FINRA 2210(d)(1)(B)",
        "policy_match": "Promissory performance language",
        "span": "guaranteed to return",
        "compliant_rewrite": "The fund can lose value.",
        "tenant_id": None,
        "split": "train",
        "split_role": "synthetic",
        "hard_negative": False,
    }
    text = render_example(row)
    assert "guaranteed to return" in text
    assert "Tenant: base" in text
    rows = [
        row,
        {**row, "message": "fine", "label": "compliant", "hard_negative": True, "span": None, "rule": None, "policy_match": None},
        {**row, "message": "tenant", "tenant_id": "northline"},
    ]
    shared = select_rows(rows, {"data": {"include_hard_negatives": False}})
    assert all(item.get("tenant_id") in (None, "base") for item in shared)
    assert all(not item["hard_negative"] for item in shared)
    configs = [load_config(path) for path in ablation_configs()]
    ranks = sorted(config["lora"]["r"] for config in configs)
    assert ranks == [8, 16, 16, 32]
    assert any(config["data"]["include_hard_negatives"] is False for config in configs)
    assert all(config["seed"] == 2210 for config in configs)
