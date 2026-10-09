"""Every comparison method that can appear in a report is defined in full in the glossary."""
import itertools
import re
from types import SimpleNamespace

from verity_rules.cli import _METHOD_GLOSSARY
from verity_rules.rule_learner import COMPARATOR_VARIANTS, comparison_method


def _reachable_methods():
    found = set()
    for variant, emb, sem, phys, inferred in itertools.product(
            COMPARATOR_VARIANTS, (False, True),
            ("IDENTIFIER", "NAME", "STREET_ADDRESS", "DESCRIPTIVE", "CATEGORICAL"),
            ("STRING", "NUMBER", "DATE", "EMAIL"), (None, "EMAIL", "DATE", "STRING")):
        prof = SimpleNamespace(semantic_type=sem, physical_kind=phys, inferred_kind=inferred)
        found.add(comparison_method(prof, emb, variant))
    return found


def test_every_reachable_method_is_defined():
    assert _reachable_methods() <= set(_METHOD_GLOSSARY)


def test_definitions_do_not_refer_to_other_methods():
    for name, text in _METHOD_GLOSSARY.items():
        others = set(_METHOD_GLOSSARY) - {name}
        for other in others:
            assert not re.search(rf"(?<![\w]){re.escape(other)}(?![\w])", text), (name, other)
        assert not re.search(r"\b(same as|described under|see )\b", text, re.I), name
