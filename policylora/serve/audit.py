"""SQLite evidence log. One row per verdict, including fail-closed."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from policylora.schema import ValidateResponse


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    tenant_id TEXT,
                    rulepack TEXT,
                    author_type TEXT,
                    verdict TEXT NOT NULL,
                    rule_cited TEXT,
                    policy_match TEXT,
                    adapter_version TEXT,
                    stage TEXT NOT NULL,
                    failure_kind TEXT,
                    latency_ms REAL,
                    response_json TEXT NOT NULL
                )
                """
            )

    def write(
        self,
        *,
        evidence_id: str,
        created_at: str,
        input_hash: str,
        tenant_id: str,
        rulepack: str,
        author_type: str,
        response: ValidateResponse,
    ) -> None:
        first = response.violations[0] if response.violations else None
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO evidence (
                    evidence_id, created_at, input_hash, tenant_id, rulepack, author_type,
                    verdict, rule_cited, policy_match, adapter_version, stage, failure_kind,
                    latency_ms, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    created_at,
                    input_hash,
                    tenant_id,
                    rulepack,
                    author_type,
                    response.verdict.value,
                    first.rule_cited if first else None,
                    first.policy_match if first else None,
                    response.adapter_version,
                    response.stage,
                    response.failure_kind,
                    response.latency_ms,
                    response.model_dump_json(),
                ),
            )

    def get(self, evidence_id: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["response_json"] = json.loads(payload["response_json"])
        return payload
