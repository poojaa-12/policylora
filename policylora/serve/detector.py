"""Detectors: mock, scripted (tests), and vLLM OpenAI-compatible."""

from __future__ import annotations

import json
from typing import Callable

import httpx

from policylora.schema import Detection
from policylora.serve.heuristic import template_rewrite, tenant_detection
from policylora.serve.prompts import DETECTION_SCHEMA, REWRITE_SCHEMA, detection_messages, rewrite_messages
from policylora.serve.registry import AdapterPin


class DetectorFailure(Exception):
    kind = "model_error"


class DetectorTimeout(DetectorFailure):
    kind = "timeout"


class DetectorMalformed(DetectorFailure):
    kind = "malformed"


class MockDetector:
    """Local stand-in. Tenant hits come from written heuristics, not from a LoRA."""

    def detect(self, body: str, tenant_id: str, adapter: AdapterPin) -> Detection:
        del adapter
        return tenant_detection(body, tenant_id)

    def rewrite(self, body: str, tenant_id: str, adapter: AdapterPin, violations: list) -> str:
        del tenant_id, adapter
        spans = [violation.span for violation in violations if getattr(violation, "span", None)]
        return template_rewrite(body, spans)


class ScriptedDetector:
    def __init__(
        self,
        detection: Detection | Callable[[str], Detection] | None = None,
        rewrite_text: str | Callable[[str], str] = "Rewritten.",
        error: Exception | None = None,
        rewrite_error: Exception | None = None,
        delay_s: float = 0.0,
    ):
        self._detection = detection or Detection(violation=False, category="compliant", confidence=0.9)
        self._rewrite = rewrite_text
        self._error = error
        self._rewrite_error = rewrite_error
        self.delay_s = delay_s
        self.detect_calls: list[str] = []
        self.rewrite_calls: list[str] = []

    def detect(self, body: str, tenant_id: str, adapter: AdapterPin) -> Detection:
        del tenant_id, adapter
        self.detect_calls.append(body)
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        if self._error:
            raise self._error
        if callable(self._detection):
            return self._detection(body)
        return self._detection

    def rewrite(self, body: str, tenant_id: str, adapter: AdapterPin, violations: list) -> str:
        del tenant_id, adapter, violations
        self.rewrite_calls.append(body)
        if self._rewrite_error:
            raise self._rewrite_error
        if callable(self._rewrite):
            return self._rewrite(body)
        return self._rewrite


def parse_detection(payload: dict, source_text: str) -> Detection:
    try:
        detection = Detection.model_validate(payload)
    except Exception as exc:
        raise DetectorMalformed("detection schema") from exc
    if detection.violation:
        if not detection.span or detection.span not in source_text:
            raise DetectorMalformed("span")
        if not detection.category or not detection.rule:
            raise DetectorMalformed("citation")
    return detection


def parse_rewrite(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        raise DetectorMalformed("empty rewrite")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, dict) and isinstance(payload.get("rewritten"), str) and payload["rewritten"].strip():
        return payload["rewritten"].strip()
    raise DetectorMalformed("rewrite json")


class VLLMDetector:
    def __init__(self, base_url: str, timeout_s: float = 2.0, policy_lookup: Callable[[str], str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.policy_lookup = policy_lookup or (lambda tenant_id: f"Tenant {tenant_id} policy.")
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s))

    def detect(self, body: str, tenant_id: str, adapter: AdapterPin) -> Detection:
        payload = self._chat(
            adapter.module_name,
            detection_messages(body, tenant_id, self.policy_lookup(tenant_id)),
            max_tokens=160,
            schema=DETECTION_SCHEMA,
            schema_name="detection",
        )
        return parse_detection(payload, body)

    def rewrite(self, body: str, tenant_id: str, adapter: AdapterPin, violations: list) -> str:
        reasons = "\n".join(
            f"- {violation.policy_match}: {violation.explanation}" for violation in violations
        )
        content = self._chat_content(
            adapter.module_name,
            rewrite_messages(body, tenant_id, self.policy_lookup(tenant_id), reasons),
            max_tokens=256,
            schema=REWRITE_SCHEMA,
            schema_name="rewrite",
        )
        return parse_rewrite(content)

    def _chat(self, model: str, messages: list[dict], max_tokens: int, schema: dict, schema_name: str) -> dict:
        content = self._chat_content(model, messages, max_tokens, schema, schema_name)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise DetectorMalformed("json") from exc

    def _chat_content(self, model: str, messages: list[dict], max_tokens: int, schema: dict, schema_name: str) -> str:
        request = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        }
        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=request)
        except httpx.TimeoutException as exc:
            raise DetectorTimeout("timeout") from exc
        except httpx.HTTPError as exc:
            raise DetectorFailure("transport") from exc
        if response.status_code != 200:
            raise DetectorFailure(f"status {response.status_code}")
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DetectorMalformed("envelope") from exc
