from __future__ import annotations

from policylora.schema import AuthorType, Detection, Violation
from policylora.serve.heuristic import cedar_span, harbor_span, northline_span
from policylora.serve.merge import Signal, merge_signals
from policylora.serve.rules import RulesEngine


def test_rules_ignore_negated_risk_free_and_catch_promise():
    rules = RulesEngine()
    clean = rules.evaluate("This investment is not risk-free and you can lose money.")
    assert clean.verdict == "pass"
    hit = rules.evaluate("This note is a risk-free way to beat inflation.")
    assert hit.verdict == "rewrite"
    assert hit.violations[0].category == "promissory_claim"


def test_rules_do_not_treat_can_lose_as_cant_lose():
    rules = RulesEngine()
    assert rules.evaluate("You can lose money, including principal.").verdict == "pass"
    assert rules.evaluate("You can't lose money in this fund.").verdict == "rewrite"


def test_restricted_product_blocks_and_policy_text_does_not():
    rules = RulesEngine()
    blocked = rules.evaluate("I recommend the Apex Growth Fund for your portfolio this week.")
    assert blocked.verdict == "block"
    assert blocked.violations[0].policy_match == "Restricted product policy"
    allowed = rules.evaluate("Do not recommend the Apex Growth Fund to retail clients.")
    assert allowed.verdict == "pass"


def test_options_disclosure_rule():
    rules = RulesEngine()
    missing = rules.evaluate("I would buy calls on the index before earnings.")
    assert any(item.category == "options_without_risk_disclosure" for item in missing.violations)
    present = rules.evaluate("I would buy calls on the index. You can lose more than you invest.")
    assert all(item.category != "options_without_risk_disclosure" for item in present.violations)


def test_merge_block_beats_pass():
    block = Signal(
        "block",
        [
            Violation(
                rule_cited="Firm restricted-product policy",
                policy_match="Restricted product policy",
                explanation="blocked",
                span="Apex Growth Fund",
                confidence=1,
                category="restricted_product",
            )
        ],
        "rules",
    )
    ok = Signal("pass", [], "slm")
    verdict, violations = merge_signals([block, ok], AuthorType.ai)
    assert verdict == "block"
    assert violations[0].category == "restricted_product"


def test_human_rewrite_becomes_block():
    signal = Signal(
        "rewrite",
        [
            Violation(
                rule_cited="FINRA 2210(d)(1)(B)",
                policy_match="Promissory performance language",
                explanation="promise",
                span="guaranteed to return",
                confidence=1,
                category="promissory_claim",
            )
        ],
        "rules",
    )
    verdict, _ = merge_signals([signal, Signal("pass", [], "slm")], AuthorType.human)
    assert verdict == "block"


def test_tenant_heuristics_are_tenant_specific():
    text = "The fund returned 11% last year. Past performance does not guarantee future results."
    assert northline_span(text)
    assert cedar_span(text) is None
    fees = "The expense ratio is 0.45%."
    assert northline_span(fees) is None
    options = "Consider listed options. You can lose more than you invest."
    assert harbor_span(options) is None
    assert harbor_span("Consider listed options before earnings.")
    assert cedar_span("We never say the word guaranteed in ads.")
    assert cedar_span("Brokerage cash is not FDIC insured.")
    disclosure = Detection(violation=False, category="compliant", confidence=0.9)
    assert disclosure.violation is False
