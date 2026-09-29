from __future__ import annotations

from pathlib import Path

from policylora.data.catalog import assign_train_splits, build_gold, build_ood, build_tenant_eval, build_train_candidates
from policylora.data.checks import automatic_checks
from policylora.data.dedup import VectorIndex, dedupe
from policylora.data.extract import parse_excerpts
from policylora.data.generate import route_disagreement
from policylora.data.load import LeakageError, assert_trainable
from policylora.data.splits import cell_key
from policylora.paths import repo_root
from policylora.serve.rules import RulesEngine


def test_gold_is_large_and_checks_pass():
    gold = build_gold()
    assert len(gold) >= 200
    assert len({row["id"] for row in gold}) == len(gold)
    assert all(not automatic_checks(row) for row in gold)


def test_holdouts_do_not_share_ids():
    ids = [row["id"] for row in build_gold() + build_ood() + build_tenant_eval()]
    assert len(ids) == len(set(ids))


def test_rules_match_obvious_gold_and_miss_a_subtle_case():
    rules = RulesEngine()
    gold = build_gold()
    misses = []
    subtle_misses = 0
    for row in gold:
        outcome = rules.evaluate(row["message"])
        predicted = "compliant" if outcome.verdict == "pass" else outcome.violations[0].category
        if row["hard_negative"]:
            assert outcome.verdict == "pass"
        elif row["subtlety"] == "obvious" and row["label"] != "compliant":
            categories = {item.category for item in outcome.violations}
            if row["label"] not in categories:
                misses.append(row["message"])
        elif row["subtlety"] == "subtle" and outcome.verdict == "pass":
            subtle_misses += 1
    assert misses == []
    assert subtle_misses >= 1


def test_reserved_cell_is_entirely_test():
    rows = assign_train_splits(build_train_candidates())
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row["cell"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    reserved = "promissory_claim|social|retail|etf|ai|subtle|clean|base"
    assert groups[reserved] == {"test"}
    assert cell_key(next(row for row in rows if row["cell"] == reserved)) == reserved


def test_dedupe_drops_a_near_copy_and_embedding_index_drops_identical_vectors():
    gold = build_gold()[0]
    clone = dict(gold)
    clone["id"] = "clone"
    kept = dedupe([clone], [(gold["label"], gold["message"])], threshold=0.9)
    assert kept == []

    class Same:
        def __call__(self, text: str) -> list[float]:
            return [1.0, 0.0] if "same" in text else [0.0, 1.0]

    index = VectorIndex(Same())
    rows = [
        {"label": "promissory_claim", "message": "same alpha"},
        {"label": "promissory_claim", "message": "same beta"},
        {"label": "promissory_claim", "message": "other"},
    ]
    assert len(dedupe(rows, [], threshold=0.99, index=index)) == 2


def test_train_loader_rejects_holdout_ids():
    row = {"id": "gold-0000", "split": "train", "split_role": "synthetic"}
    try:
        assert_trainable([row], {"gold-0000"})
        raise AssertionError("expected leakage")
    except LeakageError:
        pass


def test_unreviewed_disagreement_goes_to_review():
    rules = RulesEngine()
    row = {"label": "compliant", "message": "The fund is guaranteed to return 8% a year.", "reviewed": False}
    assert route_disagreement(row, rules) == "review"
    row["reviewed"] = True
    assert route_disagreement(row, rules) == "keep"


def test_excerpt_parser_reads_shipped_rules():
    text = (repo_root() / "policylora/data/sources/finra_excerpts.md").read_text()
    records = parse_excerpts(text)
    assert len(records) == 4
    assert records[0]["url"].startswith("https://www.finra.org/")
    assert "fair and balanced" in records[0]["quote"]
