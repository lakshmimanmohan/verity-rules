"""Record id choice in the CLI: valid id, no id (row numbers), blanks, --id-column, _row clash.
All data is built here; no git-ignored files."""
import csv
import json

import pytest

from verity_rules.cli import main

FIRSTS = ["Ann", "Bob", "Cara", "Dev", "Eli", "Fay", "Gus", "Hal"]
NOTE = "No unique id column was found. Row numbers ({}) are used as record ids in the pairs files."


def _rows(n=60):
    """Each person appears twice (a duplicate with a small typo), so no content column is unique."""
    out = []
    for i in range(n):
        p = i // 2
        first, last = FIRSTS[p % len(FIRSTS)], f"Name{p}"
        out.append({"given": first + ("x" if i % 2 else ""), "family": last,
                    "mail": f"{first}.{last}@example.com".lower(),
                    "city": f"Town{p % 7}x{p}"})
    return out


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _learn(tmp_path, rows, *extra):
    src, out = tmp_path / "in.csv", tmp_path / "out"
    _write(src, rows)
    code = main(["learn", str(src), "--out", str(out), *extra])
    return code, out


def _rules(out):
    return json.loads((out / "rules.json").read_text(encoding="utf-8"))


def _pair_ids(out):
    with (out / "calibration_pairs.csv").open(encoding="utf-8") as fh:
        return [(r["record_id_a"], r["record_id_b"]) for r in csv.DictReader(fh)]


def test_no_id_column_uses_row_numbers_and_keeps_email(tmp_path, capsys):
    code, out = _learn(tmp_path, _rows())
    assert code == 0
    rules = _rules(out)
    assert rules["record_id_column"] == "_row"
    assert "mail" in rules["columns"]
    assert NOTE.format("_row") in rules["notes"]
    assert NOTE.format("_row") in (out / "rules_report.md").read_text(encoding="utf-8")
    assert "Warning: " + NOTE.format("_row") in capsys.readouterr().err
    ids = _pair_ids(out)
    assert ids and all(1 <= int(a) <= 60 and 1 <= int(b) <= 60 for a, b in ids)


def test_valid_id_is_used_and_no_note(tmp_path, capsys):
    rows = [{"rec_key": f"R{i:03d}", **r} for i, r in enumerate(_rows())]
    code, out = _learn(tmp_path, rows)
    assert code == 0
    rules = _rules(out)
    assert rules["record_id_column"] == "rec_key"
    assert not any("Row numbers" in n for n in rules["notes"])
    assert "Warning" not in capsys.readouterr().err
    assert all(a.startswith("R") and b.startswith("R") for a, b in _pair_ids(out))


def test_id_with_blanks_is_not_used(tmp_path):
    rows = [{"rec_key": "" if i == 5 else f"R{i:03d}", **r} for i, r in enumerate(_rows())]
    code, out = _learn(tmp_path, rows)
    assert code == 0
    rules = _rules(out)
    assert rules["record_id_column"] == "_row"
    assert NOTE.format("_row") in rules["notes"]


def test_id_column_option_valid(tmp_path):
    rows = [{"rec_key": f"R{i:03d}", "alt": f"A{i:03d}", **r} for i, r in enumerate(_rows())]
    code, out = _learn(tmp_path, rows, "--id-column", "alt")
    assert code == 0
    assert _rules(out)["record_id_column"] == "alt"
    assert all(a.startswith("A") for a, _ in _pair_ids(out))


@pytest.mark.parametrize("name, fragment", [
    ("nope", "no such column: nope"),
    ("family", "family cannot be the record id because only"),
    ("blank_col", "blank_col cannot be the record id because it has 1 blank value"),
])
def test_id_column_option_invalid(tmp_path, capsys, name, fragment):
    rows = [{"blank_col": "" if i == 0 else f"B{i}", **r} for i, r in enumerate(_rows())]
    code, out = _learn(tmp_path, rows, "--id-column", name)
    err = capsys.readouterr().err.strip()
    assert code == 1
    assert err.startswith("error: --id-column: ") and fragment in err
    assert "\n" not in err
    assert not out.exists()


def test_existing_row_column_is_not_overwritten(tmp_path):
    rows = [{"_row": "same", **r} for r in _rows()]
    code, out = _learn(tmp_path, rows)
    assert code == 0
    rules = _rules(out)
    assert rules["record_id_column"] == "_row_2"
    assert NOTE.format("_row_2") in rules["notes"]
    assert all(1 <= int(a) <= 60 for a, _ in _pair_ids(out))
