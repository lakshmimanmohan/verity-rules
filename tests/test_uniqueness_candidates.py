"""uniqueness_ratio is over non-null values; the no-candidates note and CLI warning."""
import csv
import json
import random

from verity_rules.cli import main
from verity_rules.core import NO_CANDIDATES_NOTE, learn_rules
from verity_rules.rule_learner import column_uniqueness_and_nulls


def test_uniqueness_ignores_nulls():
    values = ["a", "b", "c", "d", None, "", None, ""]
    uq, nr = column_uniqueness_and_nulls(values, len(values))
    assert uq == 1.0 and nr == 0.5
    assert column_uniqueness_and_nulls([None, ""], 2) == (0.0, 1.0)


def _no_candidate_rows(n=150):
    rng = random.Random(5)
    rows = []
    for i in range(n):
        rows.append({"rec_key": f"R{i:04d}",
                     "a": "same" if i < 50 else f"a{rng.random():.8f}",
                     "b": "same" if i < 50 else f"b{rng.random():.8f}"})
    return rows


def test_no_candidates_note_and_cli_warning(tmp_path, capsys):
    result = learn_rules(_no_candidate_rows())
    assert result["run"]["n_candidate_pairs"] == 0
    assert NO_CANDIDATES_NOTE in result["notes"]
    src = tmp_path / "in.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rec_key", "a", "b"])
        w.writeheader()
        w.writerows(_no_candidate_rows())
    assert main(["learn", str(src), "--out", str(tmp_path / "o")]) == 0
    assert "Warning: No candidate pairs" in capsys.readouterr().err
    assert NO_CANDIDATES_NOTE in json.loads((tmp_path / "o" / "rules.json").read_text())["notes"]
