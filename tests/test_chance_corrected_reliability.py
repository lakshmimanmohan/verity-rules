"""Chance-corrected reliability in learn_reliability_v2, and v1 still selectable."""
import random

import pytest

from verity_rules.core import learn_rules
from verity_rules.rule_learner import (
    ColumnProfile, RANDOM_MEAN_CEILING, learn_reliability_v2,
)


def _profile(name, rows, uniq=1.0):
    return ColumnProfile(name=name, semantic_type="DESCRIPTIVE", physical_kind="STRING",
                         values=[r[name] for r in rows], uniqueness_ratio=uniq)


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _lo_hi(profiles):
    return {p.name: (None, None) for p in profiles}


def test_formula_matches_hand_calculation():
    rng = random.Random(1)
    n = 120
    rows = []
    for i in range(n):
        m = i // 2
        rows.append({"a": f"key{m}", "b": f"id{m}",
                     "c": f"word{m % 7}" if i % 2 == 0 else f"word{rng.randrange(7)}"})
    profiles = [_profile("a", rows), _profile("b", rows), _profile("c", rows, uniq=0.1)]
    anchors = [(2 * m, 2 * m + 1) for m in range(n // 2)]
    rel, info = learn_reliability_v2(anchors, rows, profiles, _lo_hi(profiles), min_agree=1,
                                     random_pairs=_pairs(n))
    for name in ("a", "b", "c"):
        a, r = info["anchor_mean"][name], info["random_mean"][name]
        assert rel[name] == pytest.approx(min(1.0, max(0.0, (a - r) / (1 - r))))
    assert rel["a"] == 1.0 and rel["b"] == 1.0
    assert 0.0 <= rel["c"] < 1.0


def test_random_mean_at_ceiling_gives_zero_and_all_zero_falls_back():
    n = 100
    rows = [{"a": "x", "b": "x"} for _ in range(n)]
    profiles = [_profile("a", rows), _profile("b", rows)]
    pairs = _pairs(n)
    rel, info = learn_reliability_v2(pairs, rows, profiles, _lo_hi(profiles), min_agree=1,
                                     random_pairs=pairs)
    assert info["random_mean"]["a"] >= RANDOM_MEAN_CEILING
    assert rel == {"a": 1.0, "b": 1.0}  # every column was 0, so fall back to uniqueness only
    assert any("uniqueness only" in t for t in info["notes"])
    assert any("reliability set to 0" in t for t in info["notes"])


def test_column_that_agrees_on_random_pairs_gets_zero_when_others_do_not():
    rng = random.Random(2)
    n = 120
    rows = [{"a": f"key{i // 2}", "b": f"id{i // 2}", "same": "x"} for i in range(n)]
    for r in rows:
        r["noise"] = f"w{rng.randrange(1000)}"
    profiles = [_profile("a", rows), _profile("b", rows), _profile("same", rows, uniq=0.9)]
    anchors = [(2 * m, 2 * m + 1) for m in range(n // 2)]
    rel, _ = learn_reliability_v2(anchors, rows, profiles, _lo_hi(profiles), min_agree=1,
                                  random_pairs=_pairs(n))
    assert rel["same"] == 0.0 and rel["a"] == 1.0


def _records(n=300):
    rng = random.Random(7)
    out = []
    for i in range(n):
        m = i // 2
        out.append({"rec_id": f"R{i:04d}", "mail": f"user{m}@example.com",
                    "phone": f"555{m:07d}", "town": rng.choice(["Ayr", "Bath", "Cork", "Deal"]) + str(m % 40)})
    return out


def test_default_is_v2_and_v1_selectable():
    recs = _records()
    v2 = learn_rules(recs, debug=True)
    v1 = learn_rules(recs, reliability="v1")
    assert "reliability_debug" in v2 and "reliability_debug" not in v1
    assert "random_mean" in v2["reliability_debug"]
    for res in (v1, v2):
        assert all(0.0 <= c["reliability_score"] <= 1.0 for c in res["columns"].values())
    assert v1["columns"] != v2["columns"]
    with pytest.raises(ValueError):
        learn_rules(recs, reliability="v0")
