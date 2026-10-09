"""The hint verity check prints when reliability is not (fully) measured. Data is built here."""
import csv
import random

from verity_rules import cli
from verity_rules.cli import MAY_NOT_SUIT, main

MAY_NOT_SUIT_LINE = ("Verity may not suit this file. This usually means each entity appears in "
                     "many rows, or duplicates differ in every column.")


def _write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _duplicates(path):
    """Each entity twice with identical values in four distinctive columns: all measured."""
    rng = random.Random(5)
    rows, k = [], 0
    for e in range(150):
        vals = [f"{rng.randrange(10**9):09d}", f"user{e}x{rng.randrange(999)}@mail.example",
                f"{rng.randrange(999)} Oak Street Unit {e}", f"{rng.randrange(10**5):05d}-{e}"]
        for _ in range(2):
            k += 1
            rows.append([f"R{k:05d}", *vals])
    rng.shuffle(rows)
    _write(path, ["rec_key", "phone", "mail", "street", "zip"], rows)


def _repeating(path, n=300):
    """No column is half distinct: no candidate pairs, no anchors, every column not measured."""
    rng = random.Random(3)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli"]
    rows = []
    for i in range(n):
        f, l = rng.choice(firsts), f"Name{rng.randrange(8)}"
        rows.append([f"R{i:04d}", f, l, f"City{rng.randrange(6)}", f"{f}.{l}@example.com".lower()])
    _write(path, ["rec_key", "given", "family", "city", "mail"], rows)


def _many_per_entity(path, entities=20, per=100):
    """Each entity appears 100 times, with a distinct id on every row."""
    rng = random.Random(11)
    rows = []
    for e in range(entities):
        for j in range(per):
            rows.append([f"R{e:03d}-{j:03d}", f"Customer{e}", f"{e} Elm Street",
                         f"2024-{rng.randrange(1, 13):02d}-{rng.randrange(1, 28):02d}"])
    rng.shuffle(rows)
    _write(path, ["rec_key", "customer", "street", "order_date"], rows)


def _check_out(capsys, src):
    assert main(["check", str(src)]) == 0
    return capsys.readouterr().out


def test_check_hint_when_nothing_measured(tmp_path, capsys):
    src = tmp_path / "rep.csv"
    _repeating(src)
    out = _check_out(capsys, src).splitlines()
    i = next(k for k, ln in enumerate(out) if ln.startswith("Expected outcome:"))
    assert out[i].startswith("Expected outcome: reliability would not be measured")
    assert out[i + 1] == MAY_NOT_SUIT_LINE == MAY_NOT_SUIT


def test_check_hint_on_100_rows_per_entity(tmp_path, capsys):
    src = tmp_path / "many.csv"
    _many_per_entity(src)
    out = _check_out(capsys, src)
    assert "Expected outcome: reliability would not be measured" in out
    assert MAY_NOT_SUIT_LINE in out


def test_check_no_hint_when_measured_for_every_column(tmp_path, capsys):
    src = tmp_path / "dup.csv"
    _duplicates(src)
    out = _check_out(capsys, src)
    assert "Expected outcome: reliability can be measured for every column." in out
    assert "may not suit this file" not in out


def test_check_hint_when_only_partly_measured(tmp_path, capsys, monkeypatch):
    src = tmp_path / "dup.csv"
    _duplicates(src)
    real = cli.check_file

    def partly(records, id_column):
        r = real(records, id_column)
        r["measured"], r["unmeasured"] = r["measured"][:-1], r["measured"][-1:]
        return r

    monkeypatch.setattr(cli, "check_file", partly)
    out = _check_out(capsys, src).splitlines()
    i = next(k for k, ln in enumerate(out) if ln.startswith("Expected outcome:"))
    assert "only partly" in out[i]
    assert out[i + 1] == MAY_NOT_SUIT_LINE
