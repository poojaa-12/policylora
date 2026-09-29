"""Parallel rules + SLM, conservative merge, rewrite re-check, fail closed."""

from __future__ import annotations

import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone

from policylora.paths import input_hash
from policylora.schema import (
    CONFIDENCE_FLOOR,
    RISK_LEVEL,
    SUPPORTED_RULEPACKS,
    AuthorType,
    ValidateRequest,
    ValidateResponse,
    Verdict,
)
from policylora.serve.audit import AuditLog
from policylora.serve.detector import DetectorFailure, DetectorMalformed, DetectorTimeout
from policylora.serve.merge import Signal, detection_to_violation, detection_verdict, merge_signals
from policylora.serve.registry import AdapterRegistry, UnknownTenant
from policylora.serve.rules import RulesEngine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evidence_id() -> str:
    return "evd_" + secrets.token_hex(3)


class EnforcementService:
    def __init__(
        self,
        *,
        rules: RulesEngine,
        detector,
        registry: AdapterRegistry,
        audit: AuditLog,
        timeout_s: float = 2.0,
        id_factory=_evidence_id,
    ):
        self.rules = rules
        self.detector = detector
        self.registry = registry
        self.audit = audit
        self.timeout_s = timeout_s
        self.id_factory = id_factory

    def enforce(self, request: ValidateRequest) -> ValidateResponse:
        started = time.perf_counter()
        evidence_id = self.id_factory()
        hashed = input_hash(request.body)

        def finish(response: ValidateResponse) -> ValidateResponse:
            self.audit.write(
                evidence_id=response.evidence_id,
                created_at=_now(),
                input_hash=hashed,
                tenant_id=request.tenant_id,
                rulepack=request.rulepack,
                author_type=request.author_type.value,
                response=response,
            )
            return response

        def fail(kind: str, adapter_version: str = "none") -> ValidateResponse:
            return finish(
                ValidateResponse(
                    verdict=Verdict.block,
                    risk_level="high",
                    violations=[],
                    rewritten=None,
                    evidence_id=evidence_id,
                    adapter_version=adapter_version,
                    latency_ms=_elapsed(started),
                    stage="fail_closed",
                    failure_kind=kind,
                )
            )

        if request.rulepack not in SUPPORTED_RULEPACKS:
            return fail("unknown_rulepack")
        try:
            adapter = self.registry.resolve(request.tenant_id)
        except UnknownTenant:
            return fail("unknown_tenant")

        rules_box: dict = {}
        slm_box: dict = {}

        def run_rules() -> None:
            try:
                rules_box["value"] = self.rules.evaluate(request.body)
            except Exception as exc:  # noqa: BLE001 — any rules failure is fail-closed
                rules_box["error"] = exc

        def run_slm() -> None:
            try:
                slm_box["value"] = self.detector.detect(request.body, request.tenant_id, adapter)
            except Exception as exc:  # noqa: BLE001 — classified below
                slm_box["error"] = exc

        executor = ThreadPoolExecutor(max_workers=2)
        rules_future = executor.submit(run_rules)
        slm_future = executor.submit(run_slm)
        try:
            rules_future.result(timeout=self.timeout_s)
            slm_future.result(timeout=self.timeout_s)
        except (TimeoutError, FuturesTimeout):
            executor.shutdown(wait=False, cancel_futures=True)
            return fail("timeout", adapter.version_label)
        else:
            executor.shutdown(wait=True, cancel_futures=True)

        if "error" in rules_box:
            return fail("rules_error", adapter.version_label)
        slm_error = slm_box.get("error")
        if isinstance(slm_error, TimeoutError) or isinstance(slm_error, DetectorTimeout):
            return fail("timeout", adapter.version_label)
        if isinstance(slm_error, DetectorMalformed):
            return fail("malformed", adapter.version_label)
        if slm_error is not None:
            return fail("model_error", adapter.version_label)

        rules_outcome = rules_box["value"]
        detection = slm_box["value"]
        model_violation = detection_to_violation(detection)
        signals = [
            Signal(rules_outcome.verdict, rules_outcome.violations, "rules"),
            Signal(
                detection_verdict(detection, request.author_type),
                [model_violation] if model_violation else [],
                "slm",
            ),
        ]
        verdict, violations = merge_signals(signals, request.author_type)
        rewritten = None
        stage = "merge"

        if verdict == "rewrite" and request.author_type == AuthorType.ai:
            try:
                candidate = self.detector.rewrite(request.body, request.tenant_id, adapter, violations)
                recheck = self._recheck(candidate, request.tenant_id, adapter)
            except DetectorMalformed:
                return fail("malformed", adapter.version_label)
            except DetectorTimeout:
                return fail("timeout", adapter.version_label)
            except DetectorFailure:
                return fail("model_error", adapter.version_label)
            except Exception:
                return fail("model_error", adapter.version_label)
            if recheck is None:
                rewritten = candidate
                stage = "rewrite"
            else:
                verdict = recheck
                rewritten = None
                stage = "merge"

        return finish(
            ValidateResponse(
                verdict=Verdict(verdict),
                risk_level=RISK_LEVEL[verdict],
                violations=violations,
                rewritten=rewritten,
                evidence_id=evidence_id,
                adapter_version=adapter.version_label,
                latency_ms=_elapsed(started),
                stage=stage,
            )
        )

    def _recheck(self, candidate: str, tenant_id: str, adapter) -> str | None:
        rules_outcome = self.rules.evaluate(candidate)
        detection = self.detector.detect(candidate, tenant_id, adapter)
        if rules_outcome.verdict == "pass" and not detection.violation:
            return None
        if detection.violation and detection.confidence < CONFIDENCE_FLOOR and rules_outcome.verdict == "pass":
            return "escalate"
        return "block"


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
