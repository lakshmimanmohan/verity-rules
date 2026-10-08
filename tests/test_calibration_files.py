"""Calibration pairs files: seeded shuffle, separate key file, masking, --no-pairs."""
import csv
import json
import random
from pathlib import Path

from verity_rules.cli import main


def _write_csv(path: Path, n: int = 120) -> None:
    rng = random.Random(1)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli", "Fay", "Gus", "Hal"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rec_key", "given", "family", "mail"])
        for i in range(n):
            f, l = rng.choice(firsts), f"Name{rng.randrange(40)}"
            w.writerow([f"R{i:04d}", f, l, f"{f}.{l}@example.com".lower()])


def _learn(tmp_path, name, *extra, seed=42):
    src = tmp_path / "in.csv"
    if not src.exists():
        _write_csv(src)
    out = tmp_path / name
    assert main(["learn", str(src), "--out", str(out), "--seed", str(seed), *extra]) == 0
    return out


def _rows(path):
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_pairs_file_has_no_label_and_key_matches(tmp_path):
    out = _learn(tmp_path, "o")
    pairs, key = _rows(out / "calibration_pairs.csv"), _rows(out / "calibration_pairs_key.csv")
    assert "suggested_label" not in pairs[0]
    assert list(key[0]) == ["pair_id", "suggested_label"]
    ids = [r["pair_id"] for r in pairs]
    assert ids == [str(i) for i in range(1, len(pairs) + 1)]
    assert ids == [r["pair_id"] for r in key]
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    counts = {c: len(v) for c, v in rules["calibration_pairs"].items()}
    key_counts = {}
    for r in key:
        key_counts[r["suggested_label"]] = key_counts.get(r["suggested_label"], 0) + 1
    assert {c: n for c, n in counts.items() if n} == key_counts
    # key labels follow the pairs: the first pair row carries the key's first label
    by_cat = {c: [(p["record_id_a"], p["record_id_b"]) for p in v]
              for c, v in rules["calibration_pairs"].items()}
    for p, k in zip(pairs, key):
        assert (p["record_id_a"], p["record_id_b"]) in by_cat[k["suggested_label"]]


def test_shuffle_mixes_labels_and_is_seeded(tmp_path):
    a = _learn(tmp_path, "a")
    b = _learn(tmp_path, "b")
    labels = [r["suggested_label"] for r in _rows(a / "calibration_pairs_key.csv")]
    assert labels != sorted(labels, key=["likely_match", "unclear", "likely_non_match"].index)
    for name in ("calibration_pairs.csv", "calibration_pairs_key.csv", "rules.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_mask_columns(tmp_path):
    out = _learn(tmp_path, "m", "--mask-columns", "given, mail")
    rows = _rows(out / "calibration_pairs.csv")
    assert rows
    for r in rows:
        for c in ("given_a", "given_b", "mail_a", "mail_b"):
            assert r[c] in ("***", "")
        assert r["family_a"] != "***"
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    for pairs in rules["calibration_pairs"].values():
        for p in pairs:
            assert p["values_a"]["given"] == "***" and p["values_b"]["mail"] == "***"
            assert p["values_a"]["family"] != "***"
    plain = json.loads((_learn(tmp_path, "p") / "rules.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in plain.items() if k != "calibration_pairs"} == \
        {k: v for k, v in rules.items() if k != "calibration_pairs"}


def test_mask_unknown_column_is_an_error(tmp_path, capsys):
    src = tmp_path / "in.csv"
    _write_csv(src)
    assert main(["learn", str(src), "--out", str(tmp_path / "o"), "--mask-columns", "nope"]) == 1
    assert "no such column" in capsys.readouterr().err


def test_no_pairs(tmp_path, capsys):
    out = _learn(tmp_path, "n", "--no-pairs")
    assert not (out / "calibration_pairs.csv").exists()
    assert not (out / "calibration_pairs_key.csv").exists()
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    assert all(v == [] for v in rules["calibration_pairs"].values())
    assert "source data" not in capsys.readouterr().out


def test_notice_printed_when_pairs_written(tmp_path, capsys):
    _learn(tmp_path, "o")
    assert "contain values from the source data" in capsys.readouterr().out


def test_report_sections(tmp_path):
    out = _learn(tmp_path, "r")
    text = (out / "rules_report.md").read_text(encoding="utf-8")
    for heading in ("## Run details", "### Comparison methods", "### Pair score",
                    "### Choosing a match cutoff", "### Handling the pairs files"):
        assert heading in text
    assert "rule of thumb, not yet validated" in text
    assert "review effort" not in text and "grey zone" not in text
    assert "| Type |" not in text and "\|" not in text
    assert " weight:" not in text.split("## Columns")[1].split("**Caution")[0]
    assert "[primary_key]" not in text
    assert "Text similarity is character-based" in text
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    assert all("[" not in e["reason"] and e["rule"] for e in rules["columns_excluded"])
    assert set(rules["run"]) == {"verity_version", "seed", "n_candidate_pairs"}
