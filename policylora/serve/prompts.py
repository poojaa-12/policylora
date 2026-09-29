"""Detection and rewrite prompts shared by training and vLLM."""

from __future__ import annotations

DETECTION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "violation": {"type": "boolean"},
        "category": {"type": ["string", "null"]},
        "rule": {"type": ["string", "null"]},
        "policy_match": {"type": ["string", "null"]},
        "span": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
        "suggested_fix": {"type": ["string", "null"]},
    },
    "required": [
        "violation",
        "category",
        "rule",
        "policy_match",
        "span",
        "confidence",
        "explanation",
        "suggested_fix",
    ],
}

REWRITE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"rewritten": {"type": "string"}},
    "required": ["rewritten"],
}


def detection_messages(body: str, tenant_id: str, policy_text: str) -> list[dict[str, str]]:
    system = (
        "You enforce FINRA Rule 2210 and Rule 2220 plus the tenant policy below. "
        "Return only the JSON object. Cite a rule when you flag a violation. "
        "The span must be an exact substring of the message. "
        "Use confidence below 0.6 when the call is uncertain.\n\n"
        f"Tenant: {tenant_id}\n{policy_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body},
    ]


def rewrite_messages(body: str, tenant_id: str, policy_text: str, reasons: str) -> list[dict[str, str]]:
    system = (
        "Rewrite the message so it no longer violates FINRA Rule 2210, Rule 2220, "
        "or the tenant policy. Keep the same topic and any specific product facts "
        "that are still allowed. Do not promise results. Return JSON with a rewritten field.\n\n"
        f"Tenant: {tenant_id}\n{policy_text}\nReasons:\n{reasons}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": body},
    ]


def render_training_text(
    body: str,
    tenant_id: str,
    policy_text: str,
    assistant_json: str,
) -> str:
    messages = detection_messages(body, tenant_id, policy_text)
    parts = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{assistant_json}<|im_end|>")
    return "\n".join(parts)
