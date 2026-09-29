from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from policylora.schema import COMPLIANT_CLOSE, AuthorType, Detection, ValidateRequest
from policylora.serve.app import create_app
from policylora.serve.detector import (
    DetectorFailure,
    DetectorMalformed,
    DetectorTimeout,
    ScriptedDetector,
    VLLMDetector,
    parse_detection,
    parse_rewrite,
)
from policylora.serve.pipeline import EnforcementService
from policylora.serve.registry import AdapterRegistry
from policylora.serve.rules import RulesEngine
from policylora.serve.audit import AuditLog


def _service(tmp_path: Path, detector, timeout_s: float = 2.0) -> EnforcementService:
    counter = {"n": 0}

    def ids() -> str:
        counter["n"] += 1
        return f"evd_test{counter['n']:02d}"

    return EnforcementService(
        rules=RulesEngine(),
        detector=detector,
        registry=AdapterRegistry(),
        audit=AuditLog(tmp_path / "audit.db"),
        timeout_s=timeout_s,
        id_factory=ids,
    )


def _request(body: str, tenant: str = "base", author: str = "ai", rulepack: str = "financial_services_client_communications") -> ValidateRequest:
    return ValidateRequest(body=body, tenant_id=tenant, author_type=AuthorType(author), rulepack=rulepack)


def test_parallel_rules_block_still_calls_model(tmp_path: Path):
    detector = ScriptedDetector(Detection(violation=False, category="compliant", confidence=0.99))
    response = _service(tmp_path, detector).enforce(
        _request("I recommend the Apex Growth Fund for your portfolio.")
    )
    assert response.verdict.value == "block"
    assert response.stage == "merge"
    assert detector.detect_calls
    assert detector.rewrite_calls == []
    assert response.violations[0].policy_match == "Restricted product policy"


def test_ai_rewrite_is_rechecked(tmp_path: Path):
    def detect(body: str) -> Detection:
        if "safe way to beat" in body:
            return Detection(
                violation=True,
                category="exaggerated_unwarranted",
                rule="FINRA 2210(d)(1)(B)",
                policy_match="Unwarranted safety claim",
                span="safe way to beat",
                confidence=0.93,
                explanation="promises safety",
            )
        return Detection(violation=False, category="compliant", confidence=0.9)

    detector = ScriptedDetector(detect, rewrite_text=COMPLIANT_CLOSE)
    response = _service(tmp_path, detector).enforce(
        _request("The Harbor Index Fund is a safe way to beat the market.")
    )
    assert response.verdict.value == "rewrite"
    assert response.stage == "rewrite"
    assert response.rewritten == COMPLIANT_CLOSE
    assert "safe way to beat" not in response.rewritten
    assert len(detector.detect_calls) == 2


def test_rewrite_that_still_violates_is_not_delivered(tmp_path: Path):
    detection = Detection(
        violation=True,
        category="promissory_claim",
        rule="FINRA 2210(d)(1)(B)",
        policy_match="Promissory performance language",
        span="guaranteed to return",
        confidence=0.97,
        explanation="promise",
    )

    def detect(body: str) -> Detection:
        if "guaranteed to return" in body:
            return detection
        return Detection(violation=False, category="compliant", confidence=0.2)

    original = "The fund is guaranteed to return 8% a year."
    detector = ScriptedDetector(detect, rewrite_text=original)
    response = _service(tmp_path, detector).enforce(_request(original))
    assert response.verdict.value == "block"
    assert response.rewritten is None


def test_low_confidence_recheck_escalates(tmp_path: Path):
    def detect(body: str) -> Detection:
        if body.startswith("CLEAN"):
            return Detection(violation=True, category="promissory_claim", rule="FINRA 2210(d)(1)(B)", policy_match="x", span="CLEAN", confidence=0.4, explanation="unsure")
        return Detection(
            violation=True,
            category="promissory_claim",
            rule="FINRA 2210(d)(1)(B)",
            policy_match="Promissory performance language",
            span="guaranteed to return",
            confidence=0.9,
            explanation="promise",
        )

    # The original is a rules hit, so the first verdict is rewrite.
    # The rewrite is the word CLEAN, which the scripted model flags at low confidence
    # and the rules engine does not know. That re-check escalates.
    detector = ScriptedDetector(detect, rewrite_text="CLEAN placeholder")
    response = _service(tmp_path, detector).enforce(_request("The fund is guaranteed to return 8% a year."))
    assert response.verdict.value == "escalate"
    assert response.rewritten is None


def test_human_author_does_not_rewrite(tmp_path: Path):
    detector = ScriptedDetector(Detection(violation=False, category="compliant", confidence=0.9))
    response = _service(tmp_path, detector).enforce(
        _request("The fund is guaranteed to return 8% a year.", author="human")
    )
    assert response.verdict.value == "block"
    assert response.rewritten is None
    assert detector.rewrite_calls == []


def test_unknown_tenant_does_not_call_model(tmp_path: Path):
    detector = ScriptedDetector()
    response = _service(tmp_path, detector).enforce(_request("Hello there.", tenant="acme"))
    assert response.verdict.value == "block"
    assert response.stage == "fail_closed"
    assert response.failure_kind == "unknown_tenant"
    assert detector.detect_calls == []
    row = AuditLog(tmp_path / "audit.db").get(response.evidence_id)
    assert row["verdict"] == "block"
    assert row["stage"] == "fail_closed"


def test_unknown_rulepack_fails_closed(tmp_path: Path):
    detector = ScriptedDetector()
    response = _service(tmp_path, detector).enforce(_request("Hello there.", rulepack="hipaa"))
    assert response.failure_kind == "unknown_rulepack"
    assert detector.detect_calls == []


def test_timeout_malformed_and_model_error(tmp_path: Path):
    for index, (error, kind) in enumerate(
        (
            (DetectorTimeout("timeout"), "timeout"),
            (DetectorMalformed("span"), "malformed"),
            (DetectorFailure("down"), "model_error"),
        )
    ):
        detector = ScriptedDetector(error=error)
        response = _service(tmp_path / f"case{index}", detector).enforce(
            _request("A plain status update with no pitch.")
        )
        assert response.stage == "fail_closed"
        assert response.failure_kind == kind
        assert response.verdict.value == "block"


def test_pipeline_timeout_when_detector_hangs(tmp_path: Path):
    detector = ScriptedDetector(delay_s=0.3)
    response = _service(tmp_path, detector, timeout_s=0.05).enforce(_request("A plain status update with no pitch."))
    assert response.failure_kind == "timeout"


def test_registry_pin_and_rollback(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(Path("policylora/serve/adapters/registry.json").read_text())
    registry = AdapterRegistry(path)
    assert registry.resolve("northline").version == "v0"
    document = json.loads(path.read_text())
    document["adapters"]["northline"]["active"] = "v-1"
    path.write_text(json.dumps(document))
    rolled = AdapterRegistry(path).resolve("northline")
    assert rolled.version == "v-1"
    assert rolled.module_name == "northline-v-1"


def test_parse_detection_rejects_missing_span():
    try:
        parse_detection(
            {
                "violation": True,
                "category": "promissory_claim",
                "rule": "FINRA 2210(d)(1)(B)",
                "policy_match": "x",
                "span": "not in text",
                "confidence": 0.9,
                "explanation": "x",
                "suggested_fix": None,
            },
            "hello",
        )
        raise AssertionError("expected malformed")
    except DetectorMalformed:
        pass


def test_parse_rewrite_accepts_json_and_plain_text():
    assert parse_rewrite('{"rewritten": "Hello."}') == "Hello."
    assert parse_rewrite("Hello from the model.") == "Hello from the model."


def test_vllm_detector_sends_adapter_module_and_short_budget():
    seen = {}

    def handler(request):
        import httpx

        seen["body"] = json.loads(request.content.decode())
        content = json.dumps(
            {
                "violation": False,
                "category": "compliant",
                "rule": None,
                "policy_match": None,
                "span": None,
                "confidence": 0.8,
                "explanation": "ok",
                "suggested_fix": None,
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    import httpx

    detector = VLLMDetector("http://vllm.test/v1", timeout_s=1)
    detector._client = httpx.Client(transport=httpx.MockTransport(handler))
    from policylora.serve.registry import AdapterPin

    pin = AdapterPin(
        tenant_id="northline",
        adapter_name="northline",
        version="v0",
        module_name="northline-v0",
        path="artifacts/adapters/northline/v0",
    )
    detection = detector.detect("Hello", "northline", pin)
    assert detection.violation is False
    assert seen["body"]["model"] == "northline-v0"
    assert seen["body"]["max_tokens"] <= 200
    assert seen["body"]["temperature"] == 0


def test_api_demo_flows(tmp_path: Path):
    app = create_app(db_path=tmp_path / "audit.db")
    client = TestClient(app)

    rewrite = client.post(
        "/v1/validate",
        json={
            "body": "The Harbor Index Fund is a safe way to beat the market.",
            "tenant_id": "base",
            "author_type": "ai",
            "rulepack": "financial_services_client_communications",
        },
    )
    assert rewrite.status_code == 200
    body = rewrite.json()
    assert body["verdict"] == "rewrite"
    assert body["stage"] == "rewrite"
    assert "safe way to beat" not in body["rewritten"]
    assert body["violations"][0]["rule_cited"].startswith("FINRA")

    base = client.post(
        "/v1/validate",
        json={
            "body": "The Horizon Equity Fund returned 11% last year. Past performance does not guarantee future results.",
            "tenant_id": "base",
            "author_type": "ai",
        },
    )
    northline = client.post(
        "/v1/validate",
        json={
            "body": "The Horizon Equity Fund returned 11% last year. Past performance does not guarantee future results.",
            "tenant_id": "northline",
            "author_type": "ai",
        },
    )
    assert base.json()["verdict"] == "pass"
    assert northline.json()["verdict"] == "rewrite"
    assert northline.json()["adapter_version"].startswith("northline@")

    blocked = client.post(
        "/v1/validate",
        json={
            "body": "Yes. The Apex Growth Fund is a safe way to generate strong returns. I recommend moving 25% of your portfolio into the fund this week.",
            "tenant_id": "base",
            "author_type": "ai",
        },
    )
    assert blocked.json()["verdict"] == "block"
    assert blocked.json()["rewritten"] is None

    evidence = client.get(f"/v1/evidence/{rewrite.json()['evidence_id']}")
    assert evidence.status_code == 200
    metrics = client.get("/metrics")
    assert "policylora_validate_requests_total" in metrics.text
