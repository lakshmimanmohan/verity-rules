"""Report layout, the no-anchor reason wording, and columns.csv."""
import csv
import json
import random

import pytest

from verity_rules.cli import main

NOT_MEASURED = ("Reliability was not measured (no anchor pairs), so it is set to 1.0 "
                "and does not change the weight.")


def _repeating_csv(path, n=300):
    """A transaction-like table: no column has half of its values distinct."""
    rng = random.Random(3)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rec_key", "given", "family", "city", "mail"])
        for i in range(n):
            f, l = rng.choice(firsts), f"Name{rng.randrange(8)}"
            w.writerow([f"R{i:04d}", f, l, f"City{rng.randrange(6)}", f"{f}.{l}@example.com".lower()])


def _distinct_csv(path, n=300):
    rng = random.Random(4)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli", "Fay", "Gus", "Hal"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rec_key", "given", "family", "mail"])
        for i in range(n):
            f, l = rng.choice(firsts), f"Name{rng.randrange(40)}"
            w.writerow([f"R{i:04d}", f, l, f"{f}.{l}@example.com".lower()])


def _run(tmp_path, maker, name, *extra):
    src = tmp_path / f"{name}.csv"
    maker(src)
    out = tmp_path / f"out_{name}"
    assert main(["learn", str(src), "--out", str(out), *extra]) == 0
    return out, (out / "rules_report.md").read_text(encoding="utf-8")


def test_no_candidates_box_and_reason(tmp_path):
    out, report = _run(tmp_path, _repeating_csv, "rep")
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    assert rules["run"]["n_candidate_pairs"] == 0
    lines = report.splitlines()
    assert lines[2] == "> **Read this first**"
    box = " ".join(ln for ln in lines[2:7] if ln.startswith(">"))
    assert "no column has at least half of its values distinct" in box
    assert "Reliability was not measured" in box and "distinctiveness only" in box
    for c in rules["columns"].values():
        assert NOT_MEASURED in c["reason"]
        assert "agrees" not in c["reason"] and c["reliability_score"] == 1.0
    assert "Reliability was not measured, so the weights come from distinctiveness only." in report


def test_box_absent_when_candidates_exist(tmp_path):
    _, report = _run(tmp_path, _distinct_csv, "dist")
    assert "Read this first" not in report


def test_table_columns_and_details(tmp_path):
    _, report = _run(tmp_path, _distinct_csv, "dist")
    assert "| Column | Weight | Reliability | Compared by | Reason |" in report
    assert "### Details" in report and "| Column | Anchors |" in report
    para = report.split("\n\n")[1]
    assert para.startswith("Verity analysed 300 rows and kept ")
    assert "heaviest" in para


def test_columns_csv(tmp_path):
    out, _ = _run(tmp_path, _distinct_csv, "dist")
    with (out / "columns.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    assert [r["column"] for r in rows] == sorted(
        rules["columns"], key=lambda c: (-rules["columns"][c]["composite_weight"], c))
    for r in rows:
        c = rules["columns"][r["column"]]
        for f in ("semantic_type", "comparison_method", "reason"):
            assert r[f] == c[f]
        for f in ("uniqueness_ratio", "null_rate", "reliability_score", "composite_weight"):
            assert float(r[f]) == c[f]
        assert r["anchors"] != ""


def test_columns_csv_in_overwrite_guard(tmp_path, capsys):
    out, _ = _run(tmp_path, _distinct_csv, "dist")
    for f in out.iterdir():
        if f.name != "columns.csv":
            f.unlink()
    assert main(["learn", str(tmp_path / "dist.csv"), "--out", str(out)]) == 1
    assert "columns.csv" in capsys.readouterr().err
