"""Report layout and guidance wording. Data is built in each test."""
import csv
import random

from verity_rules.cli import main

CAUTION = "**Caution:**"
COMPARE = ("Compare these weights with what you know about the data before using them. A column "
           "that is generally a weak identifier can outrank a stronger one if its values are "
           "more distinct here.")
CLOSEST = ("If your tool does not have this comparator, use its closest one: edit distance or "
           "Jaro-Winkler for text, exact match for ids. For email, compare the part before the @ "
           "and the domain separately if you can. The weights stay valid. Set the cutoff using "
           "the comparator you actually use, because a cutoff only makes sense for the scoring "
           "it was chosen with.")
NO_MATCHES = ("This run found no likely matches, so calibration_pairs.csv cannot be used to "
              "choose a cutoff. Add known matches from your own data. To set a cutoff, label "
              "pairs from your own data and follow steps 2 to 6 below.")


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


def _low_confidence(path):
    """Duplicates agree on only two columns, so each pair has one agreeing other column."""
    rng = random.Random(7)
    rows, k = [], 0
    for e in range(200):
        phone, mail = f"{rng.randrange(10**9):09d}", f"user{e}x{rng.randrange(999)}@mail.example"
        for _ in range(2):
            k += 1
            rows.append([f"R{k:05d}", phone, mail, f"{rng.randrange(10**9):09d}-w{k}",
                         f"Name {rng.randrange(10**6)} {k}"])
    rng.shuffle(rows)
    _write(path, ["rec_key", "phone", "mail", "b", "c"], rows)


def _repeating(path, n=300):
    """No column is half distinct: no candidate pairs, no anchors, every column not measured."""
    rng = random.Random(3)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli"]
    rows = []
    for i in range(n):
        f, l = rng.choice(firsts), f"Name{rng.randrange(8)}"
        rows.append([f"R{i:04d}", f, l, f"City{rng.randrange(6)}", f"{f}.{l}@example.com".lower()])
    _write(path, ["rec_key", "given", "family", "city", "mail"], rows)


def _report(tmp_path, maker, name):
    src = tmp_path / f"{name}.csv"
    maker(src)
    out = tmp_path / f"out_{name}"
    assert main(["learn", str(src), "--out", str(out)]) == 0
    return (out / "rules_report.md").read_text(encoding="utf-8")


def test_caution_directly_under_weights_table_then_compare_sentence(tmp_path):
    report = _report(tmp_path, _duplicates, "dup")
    lines = report.split("## Columns, by weight", 1)[1].splitlines()
    last_row = max(i for i, ln in enumerate(lines[:lines.index("### Details")])
                   if ln.startswith("| "))
    block = [ln for ln in lines[last_row + 1:] if ln.strip()][:3]
    assert lines[last_row + 1] == "" and lines[last_row + 2].startswith(CAUTION)
    assert block[0].startswith(CAUTION)
    assert block[1] == COMPARE
    assert block[2] == "### Details"
    assert report.count(CAUTION) == 1


def test_markers_not_defined_when_no_column_carries_them(tmp_path):
    report = _report(tmp_path, _duplicates, "dup")
    assert "(low confidence)" not in report and "(not measured)" not in report
    assert "**Low confidence**" not in report and "**Not measured**" not in report


def test_low_confidence_defined_only_when_present(tmp_path):
    report = _report(tmp_path, _low_confidence, "low")
    assert "(low confidence)" in report and "(not measured)" not in report
    assert "**Low confidence** marks" in report
    assert "**Not measured**" not in report


def test_not_measured_defined_only_when_present(tmp_path):
    report = _report(tmp_path, _repeating, "rep")
    assert "(not measured)" in report and "(low confidence)" not in report
    assert "**Not measured** marks" in report
    assert "**Low confidence**" not in report


def test_comparison_method_guidance_in_how_to_use(tmp_path):
    report = _report(tmp_path, _duplicates, "dup")
    how = report.split("## How to use these rules", 1)[1].split("### Pair score", 1)[0]
    assert CLOSEST in " ".join(how.split())


def test_cutoff_section_is_short_when_no_likely_matches(tmp_path):
    report = _report(tmp_path, _repeating, "rep")
    body = report.split("### Choosing a match cutoff", 1)[1].split("\n## ", 1)[0]
    body = body.split("### Handling the pairs files")[0]
    assert body.lstrip("\n").startswith(NO_MATCHES)
    assert "Verity does not provide a cutoff" not in body
    assert "1. " not in body.split("2. Score")[0]
    for step in ("2. Score every", "3. Sort", "4. For each", "5. Pick", "6. If a false"):
        assert step in body


def test_cutoff_section_full_when_likely_matches_exist(tmp_path):
    report = _report(tmp_path, _duplicates, "dup")
    body = report.split("### Choosing a match cutoff", 1)[1]
    assert "Verity does not provide a cutoff. To set your own:" in body
    assert "1. Open `calibration_pairs.csv`" in body
    assert "To set a cutoff, label pairs" not in report
