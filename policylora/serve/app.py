"""FastAPI enforcement API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from policylora.paths import repo_root
from policylora.schema import ValidateRequest, ValidateResponse
from policylora.serve.audit import AuditLog
from policylora.serve.detector import MockDetector, VLLMDetector
from policylora.serve.pipeline import EnforcementService
from policylora.serve.registry import AdapterRegistry
from policylora.serve.rules import RulesEngine

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
logger = logging.getLogger("policylora")


def build_detector():
    mode = os.environ.get("DETECTOR", "mock")
    if mode == "mock":
        return MockDetector()
    if mode == "vllm":
        return VLLMDetector(
            base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
            timeout_s=float(os.environ.get("DETECTOR_TIMEOUT_S", "2")),
        )
    raise RuntimeError(f"Unknown DETECTOR={mode}")


def create_app(
    *,
    detector=None,
    db_path: Path | None = None,
    registry_path: Path | None = None,
    timeout_s: float | None = None,
) -> FastAPI:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    chosen = detector if detector is not None else build_detector()
    audit_path = db_path or Path(os.environ.get("AUDIT_DB", repo_root() / "artifacts" / "audit.db"))
    registry = AdapterRegistry(registry_path)
    metrics = CollectorRegistry()
    requests_total = Counter(
        "policylora_validate_requests_total",
        "Validate requests by verdict.",
        ["verdict", "tenant_id", "stage"],
        registry=metrics,
    )
    latency = Histogram(
        "policylora_validate_latency_seconds",
        "Validate handler latency.",
        ["stage"],
        registry=metrics,
    )
    service = EnforcementService(
        rules=RulesEngine(),
        detector=chosen,
        registry=registry,
        audit=AuditLog(audit_path),
        timeout_s=timeout_s if timeout_s is not None else float(os.environ.get("DETECTOR_TIMEOUT_S", "2")),
    )
    app = FastAPI(title="PolicyLoRA", version="0.1.0")
    app.state.service = service
    app.state.metrics = metrics

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "detector": type(chosen).__name__}

    @app.post("/v1/validate", response_model=ValidateResponse)
    def validate(request: ValidateRequest) -> ValidateResponse:
        response = service.enforce(request)
        requests_total.labels(response.verdict.value, request.tenant_id, response.stage).inc()
        latency.labels(response.stage).observe(response.latency_ms / 1000)
        logger.info(
            json.dumps(
                {
                    "event": "verdict",
                    "evidence_id": response.evidence_id,
                    "tenant_id": request.tenant_id,
                    "verdict": response.verdict.value,
                    "stage": response.stage,
                    "adapter_version": response.adapter_version,
                    "latency_ms": response.latency_ms,
                    "failure_kind": response.failure_kind,
                }
            )
        )
        return response

    @app.get("/v1/evidence/{evidence_id}")
    def evidence(evidence_id: str) -> dict:
        row = service.audit.get(evidence_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown evidence id")
        return row

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(generate_latest(metrics), media_type=CONTENT_TYPE_LATEST)

    return app
