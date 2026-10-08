"""Group-aware row sampling, on a small synthetic file built inside the test."""
import csv
import json
import random

from verity_rules.cli import main
from verity_rules.sampling import MAX_GROUP_ROWS, build_groups, group_sample_indices

N_PAIRS, N_SINGLES, BIG = 150, 100, 60


def _records():
    """150 duplicate pairs sharing a 'contact' value, 100 singles, and a chain of 60 rows linked
    through 'contact' (rows 2k, 2k+1) and 'alt' (rows 2k+1, 2k+2): one group over 50 rows."""
    rng = random.Random(7)
    rows = []

    def add(contact, alt):
        rows.append({"contact": contact, "alt": alt, "town": f"Town{rng.randrange(30)}",
                     "label": f"L{rng.randrange(50)}"})
    for e in range(N_PAIRS):
        for _ in range(2):
            add(f"person{e:04d}@mail.test", f"alt-p{rng.randrange(10**9)}")
    for s in range(N_SINGLES):
        add(f"single{s:04d}@mail.test", f"alt-s{s}")
    for k in range(BIG):
        add(f"chain{k // 2:04d}@mail.test", f"alt-c{(k + 1) // 2:04d}")
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["rec_key"] = f"R{i:05d}"
    return rows


def _by_contact(records, idx):
    kept = [records[i]["contact"] for i in idx]
    return {c: kept.count(c) for c in set(kept)}


def test_groups_stay_together():
    recs = _records()
    idx, info = group_sample_indices(recs, 200, seed=42)
    assert len(idx) == 200 and idx == sorted(idx)
    pairs = {r["contact"] for r in recs if r["contact"].startswith("person")}
    counts = _by_contact(recs, idx)
    for c in pairs:
        assert counts.get(c, 0) in (0, 2), c  # both rows of a pair, or neither
    assert info["groups_kept"] > 50


def test_two_runs_identical():
    recs = _records()
    assert group_sample_indices(recs, 200, seed=42) == group_sample_indices(recs, 200, seed=42)
    assert group_sample_indices(recs, 200, seed=1)[0] != group_sample_indices(recs, 200, seed=2)[0]


def test_group_over_50_rows_is_not_kept_as_a_block():
    recs = _records()
    groups = build_groups(recs, ["contact", "alt"])
    assert max(len(g) for g in groups) > MAX_GROUP_ROWS
    big = set(max(groups, key=len))
    idx, info = group_sample_indices(recs, 300, seed=42)
    assert info["oversized_groups"] == 1
    kept_big = len(big & set(idx))
    assert kept_big < len(big)  # never the whole block
    # the block's rows only enter as single rows, so they are not sampled as a connected unit
    assert len(idx) == 300


def test_cli_random_and_groups_modes(tmp_path):
    recs = _records()
    src = tmp_path / "in.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    out = {}
    for mode in ("random", "groups"):
        o = tmp_path / mode
        assert main(["learn", str(src), "--out", str(o), "--max-rows", "200", "--sample-mode", mode]) == 0
        out[mode] = json.loads((o / "rules.json").read_text(encoding="utf-8"))
        s = out[mode]["run"]["sampling"]
        assert out[mode]["n_records"] == 200 == s["rows_analysed"]
        assert s["rows_in_file"] == len(recs) and s["mode"] == mode
        report = (o / "rules_report.md").read_text(encoding="utf-8")
        assert f"Sample mode: {mode}" in report and "random sample" not in report
    assert out["groups"]["run"]["sampling"]["groups_kept"] > 0
    assert out["random"]["run"]["sampling"]["groups_kept"] == 0
    rep = (tmp_path / "groups" / "rules_report.md").read_text(encoding="utf-8")
    assert "can still be split by the sampling" in rep
