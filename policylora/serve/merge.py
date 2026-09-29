"""Conservative merge of the rules engine and the SLM."""

from __future__ import annotations

from dataclasses import dataclass

from policylora.schema import SEVERITY, AuthorType, Detection, Violation


@dataclass
class Signal:
    verdict: str
    violations: list[Violation]
    source: str


def detection_verdict(detection: Detection, author: AuthorType) -> str:
    if not detection.violation:
        return "pass"
    if detection.confidence < 0.6:
        return "escalate"
    if author == AuthorType.human:
        return "block"
    return "rewrite"


def detection_to_violation(detection: Detection) -> Violation | None:
    if not detection.violation:
        return None
    return Violation(
        rule_cited=detection.rule or "Unspecified rule",
        policy_match=detection.policy_match or detection.category or "Model flag",
        explanation=detection.explanation or "The model flagged this message.",
        span=detection.span,
        confidence=detection.confidence,
        suggested_fix=detection.suggested_fix,
        category=detection.category,
    )


def merge_signals(signals: list[Signal], author: AuthorType) -> tuple[str, list[Violation]]:
    verdict = max((signal.verdict for signal in signals), key=lambda item: SEVERITY[item])
    if verdict == "rewrite" and author == AuthorType.human:
        verdict = "block"
    violations: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for signal in signals:
        if signal.verdict == "pass":
            continue
        for violation in signal.violations:
            key = ((violation.category or violation.policy_match), (violation.span or "").lower())
            if key in seen:
                continue
            seen.add(key)
            violations.append(violation)
    return verdict, violations
