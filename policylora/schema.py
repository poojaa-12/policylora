"""Shared request, response, and dataset models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AuthorType(str, Enum):
    ai = "ai"
    human = "human"


class Verdict(str, Enum):
    pass_ = "pass"
    rewrite = "rewrite"
    block = "block"
    escalate = "escalate"


class ValidateRequest(BaseModel):
    body: str = Field(min_length=1)
    rulepack: str = "financial_services_client_communications"
    tenant_id: str = Field(min_length=1)
    author_type: AuthorType = AuthorType.ai


class Violation(BaseModel):
    rule_cited: str
    policy_match: str
    explanation: str
    span: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_fix: str | None = None
    category: str | None = None


class ValidateResponse(BaseModel):
    verdict: Verdict
    risk_level: Literal["none", "low", "medium", "high"]
    violations: list[Violation]
    rewritten: str | None = None
    evidence_id: str
    adapter_version: str
    latency_ms: float
    stage: Literal["merge", "rewrite", "fail_closed"]
    failure_kind: str | None = None


class Detection(BaseModel):
    violation: bool
    category: str | None = None
    rule: str | None = None
    policy_match: str | None = None
    span: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = ""
    suggested_fix: str | None = None


class Example(BaseModel):
    id: str
    message: str
    label: str
    rule: str | None = None
    policy_match: str | None = None
    span: str | None = None
    compliant_rewrite: str | None = None
    channel: str
    audience: str
    product: str
    author_type: AuthorType
    subtlety: Literal["obvious", "subtle"]
    noise: Literal["clean", "disclaimer_footer", "extra_chatter"]
    tenant_id: str | None = None
    tenant_violation: bool | None = None
    hard_negative: bool = False
    split_role: str
    source: str
    source_url: str | None = None
    cell: str | None = None
    split: str | None = None


SEVERITY = {"pass": 0, "rewrite": 1, "escalate": 2, "block": 3}

RISK_LEVEL = {
    "pass": "none",
    "rewrite": "high",
    "escalate": "medium",
    "block": "high",
}

CONFIDENCE_FLOOR = 0.6
SUPPORTED_RULEPACKS = frozenset({"financial_services_client_communications"})
COMPLIANT_CLOSE = (
    "This may suit some investors depending on their objectives, risk tolerance, "
    "and time horizon. Past performance does not guarantee future results. "
    "Investing involves risk, including possible loss of principal. "
    "You can lose more than you invest."
)
