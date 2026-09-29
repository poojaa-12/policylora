"""Render a markdown report from harness output."""

from __future__ import annotations


def _macro(block: dict) -> float | None:
    categories = block.get("per_category") or {}
    recalls = [item["recall"] for item in categories.values() if item.get("recall") is not None]
    if not recalls:
        return None
    return sum(recalls) / len(recalls)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(results: dict) -> str:
    rules = results["baselines"]["rules"]
    lines = [
        "# PolicyLoRA eval report",
        "",
        "Rules engine on the gold set. Model, frontier, GPU latency, and cost are unmeasured until a prediction cache is present.",
        "",
        f"- Fingerprint: `{results['fingerprint']}`",
        f"- Suite wall clock (seconds): {_fmt(results['suite_wall_clock_s'])}",
        f"- Gold rows: {results['counts']['gold']}",
        f"- OOD rows: {results['counts']['ood']}",
        f"- Tenant-eval rows: {results['counts']['tenant_eval']}",
        "",
        "## Rules baseline",
        "",
        f"- Obvious-violation recall: {_fmt(rules['obvious_violation_recall'])}",
        f"- Hard-negative false-positive rate: {_fmt(rules['hard_negative_fpr'])}",
        f"- Span accuracy on obvious violations: {_fmt(rules['span_accuracy'])}",
        f"- Tenant-policy accuracy: {_fmt(rules['tenant_policy_accuracy'])}",
        f"- Rules latency p50/p95 ms on this CPU: {_fmt(rules['latency_p50_ms'])} / {_fmt(rules['latency_p95_ms'])}",
        f"- OOD macro recall: {_fmt(_macro(rules.get('ood', {})))}",
        "",
        "| Category | Support | Precision | Recall |",
        "| --- | --- | --- | --- |",
    ]
    for name, score in rules["per_category"].items():
        lines.append(
            f"| {name} | {score['support']} | {_fmt(score['precision'])} | {_fmt(score['recall'])} |"
        )
    lines.extend(
        [
            "",
            "## Unmeasured",
            "",
            f"- Untuned base: {results['baselines']['untuned_base']['status']}",
            f"- PolicyLoRA: {results['baselines']['policylora']['status']}",
            f"- Frontier: {results['baselines']['frontier']['status']}",
            f"- Cost per 1k: {results['cost']['status']}",
            "",
        ]
    )
    return "\n".join(lines)
