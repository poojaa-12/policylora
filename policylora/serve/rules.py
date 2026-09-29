"""High-precision rules and the restricted-product list."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from policylora.paths import negated, repo_root
from policylora.schema import COMPLIANT_CLOSE, Violation

FIXES = {
    "promissory_claim": "Remove the promise. State that results are uncertain and the investment can lose value.",
    "exaggerated_unwarranted": "Remove the unwarranted safety or magnitude claim and name the risk.",
    "false_or_misleading": "Delete the absolute claim or limit it to a sourced, dated fact with the risks beside it.",
    "material_omission": "Add the material downside that the current wording leaves out.",
    "unbalanced_presentation": "Put the risks next to the benefits in the same message.",
    "performance_claim_without_basis": "Add the period, the basis, and a statement that past or projected results are not a promise.",
    "misleading_comparison": "Name the period and why the comparison is comparable, or remove it.",
    "options_risk_minimized": "State that options can expire worthless and are not appropriate for every investor.",
    "options_without_risk_disclosure": "Add that the client can lose more than they invest before discussing the trade.",
    "restricted_product": "Do not recommend this product. Offer to discuss an alternative that is on the approved list.",
}


@dataclass(frozen=True)
class Pattern:
    category: str
    regex: re.Pattern[str]
    rule: str
    policy_match: str
    honor_negation: bool = True


def _patterns() -> list[Pattern]:
    raw = [
        ("promissory_claim", r"guaranteed to (?:return|beat|double|earn)", "FINRA 2210(d)(1)(B)", "Promissory performance language", True),
        ("promissory_claim", r"can(?:not|'t) lose\b", "FINRA 2210(d)(1)(B)", "Promissory performance language", True),
        ("promissory_claim", r"\brisk[- ]free\b", "FINRA 2210(d)(1)(B)", "Promissory performance language", True),
        ("promissory_claim", r"will double your (?:money|investment)", "FINRA 2210(d)(1)(B)", "Promissory performance language", True),
        ("material_omission", r"\bno downside\b", "FINRA 2210(d)(1)(A)", "Omission of downside risk", True),
        ("exaggerated_unwarranted", r"\bsafe way to (?:beat|generate|boost)\b", "FINRA 2210(d)(1)(B)", "Unwarranted safety claim", True),
        ("false_or_misleading", r"\bnever (?:lost|loses) money\b", "FINRA 2210(d)(1)(B)", "Misleading absolute claim", False),
        ("false_or_misleading", r"\bbeaten the market every quarter\b", "FINRA 2210(d)(1)(B)", "Misleading performance claim", True),
        ("performance_claim_without_basis", r"\bprojected to (?:return|gain|earn)\b", "FINRA 2210(d)", "Performance claim without basis", True),
        ("misleading_comparison", r"\btwice the return of\b", "FINRA 2210(d)", "Misleading comparison", True),
        ("options_risk_minimized", r"\boptions are (?:safe|simple|easy)\b", "FINRA 2220", "Options risk minimized", True),
        ("unbalanced_presentation", r"\bupside without (?:the |any )?risk\b", "FINRA 2210(d)(1)(D)", "Unbalanced presentation of benefits", True),
    ]
    return [
        Pattern(category, re.compile(expression, re.I), rule, policy, honor)
        for category, expression, rule, policy, honor in raw
    ]


@dataclass
class RulesOutcome:
    verdict: str
    violations: list[Violation]


class RulesEngine:
    def __init__(self, restricted_path: Path | None = None):
        path = restricted_path or repo_root() / "policylora/data/restricted_products.yaml"
        document = yaml.safe_load(path.read_text())
        self.products = document["products"]
        self.patterns = _patterns()

    def evaluate(self, text: str) -> RulesOutcome:
        violations: list[Violation] = []
        verdict = "pass"
        for product in self.products:
            span = self._recommended(text, product)
            if span is None:
                continue
            violations.append(self._violation("restricted_product", "Firm restricted-product policy", "Restricted product policy", span, product["explanation"]))
            verdict = "block"
        for pattern in self.patterns:
            match = pattern.regex.search(text)
            if match is None:
                continue
            if pattern.honor_negation and negated(text, match.start()):
                continue
            violations.append(
                self._violation(pattern.category, pattern.rule, pattern.policy_match, match.group(0), pattern.policy_match)
            )
            if verdict != "block":
                verdict = "rewrite"
        if self._options_missing_disclosure(text):
            match = re.search(r"\bbuy (?:calls|puts)\b", text, flags=re.I)
            span = match.group(0) if match else "buy calls"
            violations.append(
                self._violation(
                    "options_without_risk_disclosure",
                    "FINRA 2220",
                    "Options communication without max-loss disclosure",
                    span,
                    "The message discusses an options trade without stating that the client can lose more than they invest.",
                )
            )
            if verdict != "block":
                verdict = "rewrite"
        return RulesOutcome(verdict=verdict, violations=_unique(violations))

    def _options_missing_disclosure(self, text: str) -> bool:
        if re.search(r"\bbuy (?:calls|puts)\b", text, flags=re.I) is None:
            return False
        return "lose more than you invest" not in text.lower()

    def _recommended(self, text: str, product: dict) -> str | None:
        names = [product["name"], *product.get("aliases", [])]
        found = next((name for name in names if name.lower() in text.lower()), None)
        if found is None:
            return None
        if re.search(rf"\bdo not (?:recommend|buy|pitch)\b.{{0,80}}{re.escape(found)}", text, flags=re.I):
            return None
        if re.search(rf"{re.escape(found)}.{{0,40}}\brestricted\b", text, flags=re.I):
            return None
        if re.search(r"\b(recommend|buy|purchase|allocate|invest|move|portfolio|returns?)\b", text, flags=re.I) is None:
            return None
        return found

    def _violation(self, category: str, rule: str, policy: str, span: str, explanation: str) -> Violation:
        fix = FIXES.get(category, COMPLIANT_CLOSE)
        return Violation(
            rule_cited=rule,
            policy_match=policy,
            explanation=explanation,
            span=span,
            confidence=1.0,
            suggested_fix=fix,
            category=category,
        )


def _unique(violations: list[Violation]) -> list[Violation]:
    seen: set[tuple[str, str]] = set()
    kept: list[Violation] = []
    for violation in violations:
        key = (violation.category or "", (violation.span or "").lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append(violation)
    return kept
