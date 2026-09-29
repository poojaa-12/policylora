"""Metric helpers. Positive class is any non-compliant label unless noted."""

from __future__ import annotations

import math


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * fraction
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def category_scores(pairs: list[tuple[str, str]]) -> dict[str, dict]:
    labels = sorted({true for true, _ in pairs if true != "compliant"})
    scores = {}
    for label in labels:
        support = sum(true == label for true, _ in pairs)
        predicted = sum(pred == label for _, pred in pairs)
        hit = sum(true == label and pred == label for true, pred in pairs)
        scores[label] = {
            "support": support,
            "precision": safe_div(hit, predicted),
            "recall": safe_div(hit, support),
        }
    return scores


def binary_counts(pairs: list[tuple[bool, bool]]) -> dict[str, float | None]:
    tp = sum(true and pred for true, pred in pairs)
    fp = sum((not true) and pred for true, pred in pairs)
    fn = sum(true and (not pred) for true, pred in pairs)
    tn = sum((not true) and (not pred) for true, pred in pairs)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "fpr": safe_div(fp, fp + tn),
    }


def span_correct(gold: str | None, predicted: str | None) -> bool:
    if not gold or not predicted:
        return False
    return gold in predicted or predicted in gold


def slm_cost_per_1k(hourly_usd: float, messages_per_hour: float) -> float:
    if messages_per_hour <= 0:
        raise ValueError("messages_per_hour")
    return hourly_usd / messages_per_hour * 1000


def frontier_cost_per_1k(
    prompt_tokens: float,
    completion_tokens: float,
    price_in_per_million: float,
    price_out_per_million: float,
) -> float:
    per_message = (prompt_tokens / 1_000_000) * price_in_per_million + (completion_tokens / 1_000_000) * price_out_per_million
    return per_message * 1000
