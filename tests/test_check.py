"""verity check: reports what learn would find, writes nothing."""
import csv
import random

from verity_rules.cli import check_outcome, main


def _write(path, header, rows, delimiter=","):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(header)
        w.writerows(rows)


def _duplicates(path, entities=150, delimiter=","):
    """Every entity appears twice with identical values in four distinctive columns."""
    rng = random.Random(5)
    rows, k = [], 0
    for e in range(entities):
        vals = [f"{rng.randrange(10**9):09d}", f"user{e}x{rng.randrange(999)}@mail.example",
                f"{rng.randrange(999)} Oak Street Unit {e}", f"{rng.randrange(10**5):05d}-{e}"]
        for _ in range(2):
            k += 1
            rows.append([f"R{k:05d}", *vals])
    rng.shuffle(rows)
    _write(path, ["rec_key", "phone", "mail", "street", "zip"], rows, delimiter)


def _transactions(path, n=2000):
    """A table that repeats each customer many times: no column is half distinct."""
    rng = random.Random(3)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli"]
    rows = []
    for i in range(n):
        f, l = rng.choice(firsts), f"Name{rng.randrange(8)}"
        rows.append([f"R{i:05d}", f, l, f"City{rng.randrange(6)}", f"{f}.{l}@example.com".lower()])
    _write(path, ["rec_key", "given", "family", "city", "mail"], rows)


def _check(capsys, *args):
    assert main(["check", *map(str, args)]) == 0
    return capsys.readouterr().out


def test_measurable_file(tmp_path, capsys):
    src = tmp_path / "dups.csv"
    _duplicates(src)
    out = _check(capsys, src)
    assert "Rows: 300" in out
    assert "Record id: rec_key" in out
    assert "  phone" in out and "blank" in out
    assert "Candidate pairs: 150" in out
    assert "Expected outcome: reliability can be measured for every column." in out
    assert "many rows per entity" not in out


def test_not_measured_and_many_rows_note(tmp_path, capsys):
    src = tmp_path / "tx.csv"
    _transactions(src)
    out = _check(capsys, src)
    assert "Candidate pairs: 0" in out
    assert "could produce candidate pairs (at least half distinct" in out and ": none" in out
    assert ("Expected outcome: reliability would not be measured, because no column has at "
            "least half of its values distinct.") in out
    assert "This looks like a table with many rows per entity, so Verity may not suit it." in out


def test_row_numbers_when_no_unique_id(tmp_path, capsys):
    src = tmp_path / "noid.csv"
    _write(src, ["a", "b"], [["x", "1"], ["x", "2"], ["y", "1"], ["y", "2"]] * 3)
    out = _check(capsys, src)
    assert "Record id: row numbers" in out
    assert "_row" not in out.split("Per column")[1].split("Columns that")[0]


def test_delimiter_and_encoding_options(tmp_path, capsys):
    src = tmp_path / "semi.csv"
    _duplicates(src, delimiter=";")
    out = _check(capsys, src, "--delimiter", "semicolon", "--encoding", "utf-8")
    assert "semicolon-delimited" in out and "Rows: 300" in out
    bad = tmp_path / "bad.csv"
    bad.write_bytes("a,b\nx,caf\xe9\ny,z\n".encode("latin-1"))
    assert main(["check", str(bad), "--encoding", "utf-8"]) == 1
    assert "not valid utf-8" in capsys.readouterr().err


def test_writes_nothing(tmp_path, capsys):
    src = tmp_path / "dups.csv"
    _duplicates(src)
    before = sorted(p.name for p in tmp_path.iterdir())
    _check(capsys, src)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_partly_outcome_lists_columns():
    r = {"n_candidate_pairs": 40, "any_distinct": True, "measured": ["a", "b"], "unmeasured": ["c"]}
    line = check_outcome(r)
    assert "only partly" in line and "for a, b; not for c" in line
    r["measured"], r["unmeasured"] = [], ["a", "b", "c"]
    assert "would not be measured" in check_outcome(r)
