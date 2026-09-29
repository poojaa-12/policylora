"""Deterministic tenant policies used only by DETECTOR=mock.

These heuristics let a laptop exercise a tenant swap before a GPU adapter exists.
They are not the LoRA, and the eval report does not score them as one.
"""

from __future__ import annotations

import re

from policylora.schema import COMPLIANT_CLOSE, Detection

_PCT = r"\d+(?:\.\d+)?\s*(?:%|percent)(?!\w)"
_PERFORMANCE = re.compile(
    rf"\b(?:returned|returning|return of|returns of|gained|gain of|up|yield(?:ing)?|annual(?:ized)? return of)\s+\$?{_PCT}"
    rf"|\b{_PCT}\s+(?:annual(?:ized)?\s+)?(?:return|gain|yield)\b",
    re.I,
)
_FEE = re.compile(r"\b(expense ratio|fee|fees|bps|basis points)\b", re.I)
_RETURN_VERB = re.compile(r"\b(returned|returning|return of|returns of|gained|gain of|annual(?:ized)? return|yield)\b", re.I)
_OPTIONS = re.compile(r"\b(options?|calls?|puts?)\b", re.I)
_BANNED = re.compile(r"\b(guaranteed|risk[- ]free|safe|FDIC)\b", re.I)
_MAX_LOSS = "lose more than you invest"


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def northline_span(text: str) -> str | None:
    for sentence in _sentences(text):
        match = _PERFORMANCE.search(sentence)
        if match is None:
            continue
        if _FEE.search(sentence) and _RETURN_VERB.search(sentence) is None:
            continue
        return match.group(0)
    return None


def harbor_span(text: str) -> str | None:
    match = _OPTIONS.search(text)
    if match is None:
        return None
    if _MAX_LOSS in text.lower():
        return None
    return match.group(0)


def cedar_span(text: str) -> str | None:
    match = _BANNED.search(text)
    if match is None:
        return None
    return match.group(0)


def tenant_detection(text: str, tenant_id: str) -> Detection:
    if tenant_id == "northline":
        span = northline_span(text)
        if span:
            return _hit(
                "performance_claim_without_basis",
                "Northline return-number policy",
                "Northline Wealth internal policy",
                span,
                "Northline bans specific return numbers, including when a past-performance sentence is attached.",
            )
    elif tenant_id == "harbor":
        span = harbor_span(text)
        if span:
            return _hit(
                "options_without_risk_disclosure",
                "Harbor max-loss policy",
                "Harbor Options internal policy",
                span,
                "Harbor requires an explicit max-loss sentence on any options message.",
            )
    elif tenant_id == "cedar":
        span = cedar_span(text)
        if span:
            return _hit(
                "exaggerated_unwarranted",
                "Cedar banned-term policy",
                "Cedar Credit internal policy",
                span,
                "Cedar treats guaranteed, safe, risk-free, and FDIC as banned in brokerage copy, including negated uses.",
            )
    return Detection(
        violation=False,
        category="compliant",
        confidence=0.9,
        explanation="No tenant-policy match.",
        suggested_fix=None,
    )


def template_rewrite(body: str, spans: list[str]) -> str:
    kept: list[str] = []
    for sentence in _sentences(body):
        if any(span and span in sentence for span in spans):
            continue
        kept.append(sentence)
    if COMPLIANT_CLOSE not in " ".join(kept):
        kept.append(COMPLIANT_CLOSE)
    rewritten = " ".join(part.strip() for part in kept if part.strip())
    return re.sub(r"\s{2,}", " ", rewritten).strip()


def _hit(category: str, rule: str, policy: str, span: str, explanation: str) -> Detection:
    return Detection(
        violation=True,
        category=category,
        rule=rule,
        policy_match=policy,
        span=span,
        confidence=0.92,
        explanation=explanation,
        suggested_fix="Apply the tenant policy and keep the remaining facts.",
    )
