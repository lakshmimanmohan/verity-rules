"""Report wording when the run found no candidate pairs or no anchors. Data is built here."""
import csv
import random
import re

from verity_rules.cli import main


def _repeating_csv(path, n=300):
    """No column has half of its values distinct, so there are no candidate pairs or anchors."""
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


def _report(tmp_path, maker, name):
    src = tmp_path / f"{name}.csv"
    maker(src)
    out = tmp_path / f"out_{name}"
    assert main(["learn", str(src), "--out", str(out)]) == 0
    return (out / "rules_report.md").read_text(encoding="utf-8")


def _section(report, heading):
    return report.split(heading, 1)[1].split("\n## ", 1)[0]


def test_no_anchors_collapses_notes_and_trims_reasons(tmp_path):
    report = _report(tmp_path, _repeating_csv, "rep")
    notes = _section(report, "## Notes")
    assert "anchor pairs (need" not in notes
    assert notes.count("reliability was not measured and is set to 1.0 for every column") == 1
    assert "No candidate pairs found" in notes          # kept
    assert "likely_match: only 0 of 30 requested pairs available." in notes  # shortfall kept
    rows = [ln for ln in _section(report, "## Columns, by weight").splitlines()
            if ln.startswith("| ") and "Column" not in ln and "---" not in ln and "Anchors" not in ln]
    assert rows
    for ln in rows[:4]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) == 5:
            assert "(not measured)" in cells[2]
            assert re.fullmatch(r"\d+% unique, \d+% empty", cells[4]), cells[4]


def test_notes_unchanged_when_anchors_exist(tmp_path):
    report = _report(tmp_path, _distinct_csv, "dist")
    assert "set to 1.0 for every column" not in report


EXTENSION = ("This usually means each entity appears in many rows, or duplicates differ in every "
             "column. Run verity check on the file. Verity may not suit it.")
NO_MATCHES = ("This run found no likely matches, so calibration_pairs.csv cannot be used to "
              "choose a cutoff. Add known matches from your own data.")


def test_read_this_first_box_has_extension(tmp_path):
    report = _report(tmp_path, _repeating_csv, "rep")
    box = " ".join(ln.lstrip("> ") for ln in report.splitlines() if ln.startswith(">"))
    assert EXTENSION in box
    assert box.index("distinctiveness only.") < box.index(EXTENSION)


def test_box_and_extension_absent_with_candidates(tmp_path):
    report = _report(tmp_path, _distinct_csv, "dist")
    assert EXTENSION not in report


def test_cutoff_section_starts_with_no_matches_sentence(tmp_path):
    report = _report(tmp_path, _repeating_csv, "rep")
    body = _section(report, "### Choosing a match cutoff").lstrip("\n")
    assert body.startswith(NO_MATCHES)
    assert "Verity does not provide a cutoff." not in body.split("### Handling")[0]


def test_cutoff_sentence_absent_when_matches_exist(tmp_path):
    assert NO_MATCHES not in _report(tmp_path, _distinct_csv, "dist")
