"""Fail if a column name from the local sample datasets appears as a string literal in verity_rules/."""
import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("customers.csv", "products.csv")
# Vocabulary labels that happen to equal a column name (comparison_method value), not column references.
ALLOWED_VOCABULARY = {"email"}


def _column_names() -> set[str]:
    names: set[str] = set()
    for fname in DATASETS:
        with open(ROOT / "data" / fname, newline="", encoding="utf-8-sig") as fh:
            names.update(h.strip() for h in next(csv.reader(fh)))
    return names


def test_no_dataset_column_name_literals_in_package():
    names = _column_names()
    assert names, "no column names read from data/"
    hits = []
    for path in sorted((ROOT / "verity_rules").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.strip() in names - ALLOWED_VOCABULARY:
                    hits.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value!r}")
    assert not hits, "column-name literals found:\n" + "\n".join(hits)
