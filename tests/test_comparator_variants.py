"""comparator_variant: v4 is the default; v0 is the original behavior; v1 to v4 change how IDENTIFIER and text columns compare."""
import json
import random

import pytest

from verity_rules.core import learn_rules
from verity_rules.rule_learner import (
    COMPARATOR_VARIANTS, ColumnProfile, build_profiles, comparison_method, field_similarity,
)

NO_RANGE = (None, None)


def _prof(semantic, physical, inferred=""):
    return ColumnProfile(name="c", semantic_type=semantic, physical_kind=physical, values=[],
                         inferred_kind=inferred)


def _sim(prof, a, b, variant):
    return field_similarity(a, b, prof, NO_RANGE, comparator_variant=variant)


# ---- unit behavior of each comparator

def test_v0_identifier_is_exact_even_for_email():
    p = _prof("IDENTIFIER", "STRING", inferred="EMAIL")
    assert _sim(p, "a@x.com", "a@x.com", "v0") == 1.0
    assert _sim(p, "a@x.com", "A@X.COM", "v0") == 0.0  # clean_str compare is case sensitive
    assert comparison_method(p, False, "v0") == "exact"


@pytest.mark.parametrize("variant", ["v1", "v2", "v3", "v4"])
def test_v1_plus_identifier_email_uses_email_comparator(variant):
    p = _prof("IDENTIFIER", "STRING", inferred="EMAIL")
    assert _sim(p, "Ann.Lee@Example.com", "ann.lee@example.com", variant) == 1.0
    s = _sim(p, "ann.lee@example.com", "ann.li@example.com", variant)
    assert 0.0 < s < 1.0
    assert comparison_method(p, False, variant) == "email"


def test_v1_plus_identifier_date_uses_date_comparator():
    p = _prof("IDENTIFIER", "STRING", inferred="DATE")
    assert _sim(p, "2020-01-01", "2020-01-11", "v1") == pytest.approx(1 - 10 / 3650)
    assert comparison_method(p, False, "v1") == "date"


def test_v1_other_identifier_stays_exact():
    p = _prof("IDENTIFIER", "STRING", inferred="STRING")
    assert _sim(p, "AB-123", "AB-123", "v1") == 1.0
    assert _sim(p, "AB-123", "ab 123", "v1") == 0.0
    assert comparison_method(p, False, "v1") == "exact"


def test_v1_number_identifier_is_not_numeric_range():
    p = _prof("IDENTIFIER", "STRING", inferred="NUMBER")
    assert _sim(p, "5551234", "5551299", "v1") == 0.0


def test_v2_normalizes_case_and_punctuation():
    p = _prof("IDENTIFIER", "STRING", inferred="STRING")
    assert _sim(p, "AB-123", "ab 123", "v2") == 1.0
    assert _sim(p, "(555) 123-4567", "555.123.4567", "v2") == 1.0
    assert _sim(p, "AB-123", "AB-124", "v2") == 0.0
    assert _sim(p, "--", "--", "v2") == 1.0 and _sim(p, "--", "..", "v2") == 0.0
    assert comparison_method(p, False, "v2") == "exact_normalized"


def test_v3_identifier_bigram_jaccard():
    p = _prof("IDENTIFIER", "STRING", inferred="STRING")
    # "abc" -> {ab, bc}; "abd" -> {ab, bd}; 1 shared of 3
    assert _sim(p, "abc", "abd", "v3") == pytest.approx(1 / 3)
    assert _sim(p, "A-B-C", "abc", "v3") == 1.0
    assert _sim(p, "abc", "xyz", "v3") == 0.0
    assert _sim(p, "a", "a", "v3") == 1.0 and _sim(p, "a", "b", "v3") == 0.0
    assert comparison_method(p, False, "v3") == "bigram_jaccard"


def test_v4_text_character_uses_bigrams_and_other_variants_do_not():
    p = _prof("QUASI_IDENTIFIER", "STRING", inferred="STRING")
    # character sets of "abab" and "baba" are equal (1.0); bigrams {ab, ba} and {ba, ab} also equal; use a case that differs
    a, b = "abcd", "dcba"  # same character set, no shared bigram
    for v in ("v0", "v1", "v2", "v3"):
        assert _sim(p, a, b, v) == 1.0
        assert comparison_method(p, False, v) == "text_character"
    assert _sim(p, a, b, "v4") == 0.0
    assert comparison_method(p, False, "v4") == "text_bigram"
    named = _prof("NAME", "STRING", inferred="STRING")
    assert _sim(named, a, b, "v3") == 1.0 and _sim(named, a, b, "v4") == 0.0
    assert comparison_method(named, True, "v4") == "text_hybrid"


def test_nulls_score_zero_in_every_variant():
    p = _prof("IDENTIFIER", "STRING", inferred="STRING")
    for v in COMPARATOR_VARIANTS:
        assert _sim(p, "", "abc", v) == 0.0 and _sim(p, None, None, v) == 0.0


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="comparator_variant"):
        learn_rules(_records(), comparator_variant="v9")


def test_build_profiles_keeps_inferred_kind_for_identifier():
    recs = _records()
    profs = {p.name: p for p in build_profiles(["mail", "code"], recs)}
    assert profs["mail"].semantic_type == "IDENTIFIER" and profs["mail"].physical_kind == "STRING"
    assert profs["mail"].inferred_kind == "EMAIL"


# ---- end to end on a small synthetic file

def _records(n_pairs=40, n_single=600, seed=3):
    """Duplicate pairs share an entity but differ in case/punctuation; singletons pad the uniqueness."""
    rng = random.Random(seed)
    first = ["Ann", "Bob", "Cy", "Di", "Ed", "Flo", "Gus", "Hal", "Ivy", "Jo"]
    last = ["Lee", "Kim", "Roe", "Poe", "Fox", "Day", "Ray", "Ott", "Paz", "Yu"]
    out, k = [], 0
    for m in range(n_pairs + n_single):
        # every 20th singleton reuses the previous code, so the column is not a 100%-unique row key
        code_no = m - 1 if (m >= n_pairs and m % 20 == 0) else m
        base = {"mail": f"user{m}@example.com", "code": f"AB-{code_no:05d}",
                "name": f"{rng.choice(first)} {rng.choice(last)}"}
        out.append({"rec_id": f"R{k:04d}", **base})
        k += 1
        if m < n_pairs:
            out.append({"rec_id": f"R{k:04d}", "mail": base["mail"].upper(), "code": f"ab{m:05d}",
                        "name": base["name"]})
            k += 1
    return out


def _expected_method(col, variant):
    if col["semantic_type"] == "IDENTIFIER":
        if col["email"]:
            return "email"
        return {"v0": "exact", "v1": "exact", "v2": "exact_normalized"}.get(variant, "bigram_jaccard")
    return "text_bigram" if variant == "v4" else "text_character"


@pytest.mark.parametrize("variant", COMPARATOR_VARIANTS)
def test_each_variant_runs_reports_method_and_is_deterministic(variant):
    recs = _records()
    a = learn_rules(recs, comparator_variant=variant, debug=True)
    b = learn_rules(recs, comparator_variant=variant, debug=True)
    for r in (a, b):
        r.pop("threshold", None)  # KMeans centers can differ in the last bits; not public output
        r.pop("similarity_stats", None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    cols = a["columns"]
    assert cols["mail"]["semantic_type"] == "IDENTIFIER" and cols["code"]["semantic_type"] == "IDENTIFIER"
    assert cols["mail"]["comparison_method"] == ("exact" if variant == "v0" else "email")
    assert cols["code"]["comparison_method"] == _expected_method({"semantic_type": "IDENTIFIER", "email": False}, variant)
    assert cols["name"]["comparison_method"] == _expected_method({"semantic_type": cols["name"]["semantic_type"],
                                                                  "email": False}, variant)
    assert all(0.0 <= c["reliability_score"] <= 1.0 for c in cols.values())
    assert sum(c["composite_weight"] for c in cols.values()) == pytest.approx(1.0, abs=1e-4)


def test_default_is_v4_and_matches_explicit_v4():
    recs = _records()
    assert learn_rules(recs) == learn_rules(recs, comparator_variant="v4")
    assert learn_rules(recs) != learn_rules(recs, comparator_variant="v0")


def test_default_argument_of_field_similarity_and_method_is_v4():
    p = _prof("QUASI_IDENTIFIER", "STRING", inferred="STRING")
    assert field_similarity("abcd", "dcba", p, NO_RANGE) == 0.0
    assert comparison_method(p, False) == "text_bigram"


def test_report_glossary_defines_every_method():
    from verity_rules.cli import _METHOD_GLOSSARY
    all_methods = {"email", "exact", "exact_normalized", "bigram_jaccard", "text_bigram",
                   "text_character", "text_hybrid", "date", "numeric_range"}
    assert all_methods <= set(_METHOD_GLOSSARY)
    assert "order matters" in _METHOD_GLOSSARY["text_bigram"]
    assert "order matters" in _METHOD_GLOSSARY["bigram_jaccard"].lower()


def test_variants_change_reliability_of_formatting_variants():
    """Duplicates differ only in case and punctuation: exact misses them, v2 and v3 find them."""
    recs = _records()
    rel = {v: learn_rules(recs, comparator_variant=v)["columns"]["code"]["reliability_score"]
           for v in COMPARATOR_VARIANTS}
    assert rel["v0"] == rel["v1"]
    assert rel["v2"] > rel["v1"]
    assert rel["v3"] > rel["v1"]
