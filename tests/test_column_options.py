"""--exclude-columns / --keep-columns, the run object, and scoring with published weights."""
import csv
import json
import random
import tomllib
from pathlib import Path

import pytest

import verity_rules
from verity_rules.cli import main
from verity_rules.core import learn_rules
from verity_rules.rule_learner import (
    build_profiles, numeric_series_stats, weighted_pair_similarity,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, n: int = 150) -> None:
    rng = random.Random(3)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli", "Fay", "Gus", "Hal"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rec_key", "given", "family", "mail", "channel"])
        for i in range(n):
            f, l = rng.choice(firsts), f"Name{rng.randrange(60)}"
            w.writerow([f"R{i:04d}", f, l, f"{f}.{l}@example.com".lower(),
                        rng.choice(["web", "store", "phone"])])


def _learn(tmp_path, name, *extra):
    src = tmp_path / "in.csv"
    if not src.exists():
        _write_csv(src)
    out = tmp_path / name
    code = main(["learn", str(src), "--out", str(out), *extra])
    return code, out


def _rules(out):
    return json.loads((out / "rules.json").read_text(encoding="utf-8"))


def test_default_run_excludes_channel_by_rule(tmp_path):
    _, out = _learn(tmp_path, "d")
    rules = _rules(out)
    assert "channel" not in rules["columns"]
    entry = next(e for e in rules["columns_excluded"] if e["name"] == "channel")
    assert entry["rule"] == ["near_constant_field"]
    text = (out / "rules_report.md").read_text(encoding="utf-8")
    assert "(Verity rule. To include it anyway, use `--keep-columns channel`.)" in text


def test_keep_columns_includes_a_rule_excluded_column(tmp_path):
    code, out = _learn(tmp_path, "k", "--keep-columns", "channel")
    assert code == 0
    rules = _rules(out)
    col = rules["columns"]["channel"]
    assert col["composite_weight"] > 0 and col["reason"]
    assert all(e["name"] != "channel" for e in rules["columns_excluded"])
    assert any("channel" in n and "kept at your request" in n for n in rules["notes"])
    assert "Kept by you (--keep-columns): channel" in (out / "rules_report.md").read_text("utf-8")


def test_exclude_columns_drops_a_content_column(tmp_path):
    code, out = _learn(tmp_path, "e", "--exclude-columns", "given, family")
    assert code == 0
    rules = _rules(out)
    assert "given" not in rules["columns"] and "family" not in rules["columns"]
    entry = next(e for e in rules["columns_excluded"] if e["name"] == "given")
    assert entry["rule"] == ["user_excluded"]
    text = (out / "rules_report.md").read_text(encoding="utf-8")
    assert "- `given`: Excluded by you." in text
    assert "- `channel`:" in text and "Verity rule." in text
    assert abs(sum(c["composite_weight"] for c in rules["columns"].values()) - 1) < 1e-4


@pytest.mark.parametrize("args,message", [
    (["--keep-columns", "nope"], "--keep-columns: no such column: nope"),
    (["--exclude-columns", "given,nope,zzz"], "--exclude-columns: no such column: nope, zzz"),
    (["--keep-columns", "given", "--exclude-columns", "given"], "both list: given"),
    (["--keep-columns", "rec_key"], "record id column and cannot be kept"),
])
def test_bad_column_options_give_one_line_error(tmp_path, capsys, args, message):
    code, out = _learn(tmp_path, "x", *args)
    err = capsys.readouterr().err.strip()
    assert code == 1 and message in err and err.startswith("error: ") and "\n" not in err
    assert not out.exists()


def test_run_object_has_no_date(tmp_path):
    _, out = _learn(tmp_path, "r", "--seed", "7")
    run = _rules(out)["run"]
    assert set(run) == {"verity_version", "seed", "n_candidate_pairs"}
    assert run["seed"] == 7 and run["n_candidate_pairs"] > 0


def test_version_matches_pyproject():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert verity_rules.__version__ == meta["project"]["version"]


def test_reasons_have_no_weight_word(tmp_path):
    _, out = _learn(tmp_path, "w")
    for c in _rules(out)["columns"].values():
        assert "weight" not in c["reason"].lower()


def test_separability_uses_published_weights(tmp_path):
    """With few records every pair is sampled, so random_pair_mean can be recomputed
    from the published composite weights alone."""
    src = tmp_path / "in.csv"
    _write_csv(src, n=30)
    with src.open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    res = learn_rules(records, debug=True)
    cols = list(res["columns"])
    rows = [{c: r[c] for c in cols} for r in records]
    profiles = build_profiles(cols, rows)
    for p in profiles:
        p.composite_weight = res["columns"][p.name]["composite_weight"]
        assert res["columns"][p.name]["semantic_type"] == p.semantic_type
    lo_hi = {p.name: numeric_series_stats(p.values) for p in profiles}
    sims = [weighted_pair_similarity(rows[i], rows[j], profiles, lo_hi,
                                     weight_attr="composite_weight")
            for i in range(30) for j in range(i + 1, 30)]
    assert res["separability"]["random_pair_mean"] == pytest.approx(sum(sims) / len(sims), abs=1e-5)
