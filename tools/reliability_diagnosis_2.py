r"""
Reliability and candidate diagnosis against ground truth. Not shipped to users. Report only.

Two steps, because the old code (commit b2436de) must be imported from a git worktree:

  # 1. dump per-dataset facts for one code tree (run once per tree)
  .venv\Scripts\python.exe tools\reliability_diagnosis_2.py dump --code-root . --label new --out new.json
  git worktree add --detach <dir> b2436de
  .venv\Scripts\python.exe tools\reliability_diagnosis_2.py dump --code-root <dir> --label old --out old.json

  # 2. combine the two dumps and the best-cutoff section into one markdown report
  .venv\Scripts\python.exe tools\reliability_diagnosis_2.py report --new new.json --old old.json \
      --cutoff cutoff.md --h2 h2.json --out validation\reliability_diagnosis_2.md

Per dataset and content column the dump holds: learned reliability (learn_rules), ideal reliability
(chance-corrected, over ALL true duplicate pairs), anchor count and purity against truth at min_agree 1
and 2, and the candidate pairs (count, true-duplicate share, source columns). The ideal uses the same
formula as the package, clip((dup mean - random mean) / (1 - random mean), 0, 1), with the random mean from
the 1000 random pairs learn_rules samples (same seed, so same pairs). With the old code the learned value is
the raw anchor mean, because chance correction did not exist yet.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42
DATASETS = [("Copy C", "customers_degraded_C.csv"), ("Copy H", "customers_degraded_H.csv"),
            ("customers", "customers.csv")]


def dump(code_root: Path, label: str, datasets: list[tuple[str, str]]) -> dict:
    sys.path.insert(0, str(code_root))
    import pandas as pd
    from verity_rules.cli import load_records
    from verity_rules.column_classification import classify_columns
    from verity_rules.core import _build_candidates, learn_rules
    from verity_rules.rule_learner import (
        build_profiles, field_similarity, is_null, learn_reliability_v2, normalize_discriminative_weights,
        numeric_series_stats, record_id_from_row)
    import verity_rules
    assert Path(verity_rules.__file__).resolve().is_relative_to(code_root.resolve()), verity_rules.__file__

    out: dict = {"label": label, "code_root": str(code_root), "datasets": {}}
    for name, fn in datasets:
        records, _ = load_records(DATA / fn, 20_000, SEED)
        res = learn_rules(records, seed=SEED, debug=True)
        df = pd.DataFrame(records)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.where(pd.notna(df), None)
        cls = classify_columns(df)
        rows = df.to_dict(orient="records")
        kept = list(res["columns"])
        content = [h for h in df.columns if h in set(cls["content_columns"])]
        profiles = build_profiles(content, rows, set(cls["metadata_columns"]))
        normalize_discriminative_weights(profiles)
        lo_hi = {p.name: numeric_series_stats(p.values) for p in profiles}
        rid = res["record_id_column"]
        row_ids = [record_id_from_row(r, rid) for r in rows]
        t = pd.read_csv(DATA / "customers_truth.csv", dtype=str, keep_default_na=False)
        tmap = dict(zip(t.iloc[:, 0], t["true_entity_id"]))
        entity = [tmap[r] for r in row_ids]
        groups = defaultdict(list)
        for i, e in enumerate(entity):
            groups[e].append(i)
        dup_pairs = sorted(p for g in groups.values() if len(g) > 1 for p in itertools.combinations(g, 2))
        is_dup = lambda i, j: entity[i] == entity[j]  # noqa: E731

        # the 1000 random pairs learn_rules samples (random.seed(seed) then this loop)
        random.seed(SEED)
        n = len(rows)
        s: set = set()
        while len(s) < 1000:
            i, j = random.randrange(n), random.randrange(n)
            if i != j:
                s.add((min(i, j), max(i, j)))
        rnd_pairs = list(s)

        def mean_sim(p, pairs):
            vals = []
            for i, j in pairs:
                a, b = rows[i].get(p.name), rows[j].get(p.name)
                if not (is_null(a) or is_null(b)):
                    vals.append(field_similarity(a, b, p, lo_hi[p.name]))
            return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

        cand_map = _build_candidates(rows, profiles)
        cands = sorted(cand_map)
        by_src = Counter(v[0] for v in cand_map.values())
        src_dups = Counter(v[0] for k, v in cand_map.items() if is_dup(*k))
        uniq = {p.name: p.uniqueness_ratio for p in profiles}
        runs = {m: learn_reliability_v2(cands, rows, profiles, lo_hi, min_agree=m, relax=False)[1]
                for m in (1, 2)} if cands else None
        cols = {}
        for p in profiles:
            if p.name not in kept:
                continue
            dmean, dn = mean_sim(p, dup_pairs)
            rmean, rn = mean_sim(p, rnd_pairs)
            ideal = None if dmean is None or rmean is None else (
                0.0 if rmean >= 0.999 else min(1.0, max(0.0, (dmean - rmean) / (1 - rmean))))
            row = {"uniqueness": res["columns"][p.name]["uniqueness_ratio"],
                   "null_rate": res["columns"][p.name]["null_rate"],
                   "semantic_type": res["columns"][p.name]["semantic_type"],
                   "comparison_method": res["columns"][p.name]["comparison_method"],
                   "learned": res["columns"][p.name]["reliability_score"],
                   "weight": res["columns"][p.name]["composite_weight"],
                   "dup_mean": dmean, "dup_n": dn, "random_mean": rmean, "random_n": rn, "ideal": ideal,
                   "min_agree_used": (res.get("reliability_debug") or {}).get("min_agree_used", {}).get(p.name),
                   "anchor_count_used": (res.get("reliability_debug") or {}).get("anchor_count", {}).get(p.name),
                   "debug_random_mean": (res.get("reliability_debug") or {}).get("random_mean", {}).get(p.name),
                   "debug_anchor_mean": (res.get("reliability_debug") or {}).get("anchor_mean", {}).get(p.name)}
            for m in (1, 2):
                ap = runs[m]["anchor_pairs"][p.name] if runs else []
                row[f"anchors{m}"] = len(ap)
                row[f"purity{m}"] = (sum(is_dup(i, j) for i, j in ap) / len(ap)) if ap else None
                row[f"anchor_mean{m}"] = runs[m].get("anchor_mean", {}).get(p.name) if runs else None
            cols[p.name] = row
        out["datasets"][name] = {
            "n_records": n, "n_true_dup_pairs": len(dup_pairs), "excluded": res["columns_excluded"],
            "columns": cols, "notes": (res.get("run") or {}).get("notes", res.get("notes")),
            "n_candidates": len(cands), "cand_true_dups": sum(is_dup(*k) for k in cands),
            "cand_source_columns": sorted(p.name for p in profiles if p.uniqueness_ratio >= 0.5),
            "cand_by_source": dict(by_src), "cand_dups_by_source": dict(src_dups),
            "uniqueness_all": uniq, "separability": res["separability"]}
        print(f"[{label}] {name}: {len(cands)} candidates, {len(dup_pairs)} true dup pairs", flush=True)
    return out


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def pc(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def rel_table(d) -> list[str]:
    rows = [[c, f4(v["uniqueness"]), f4(v["learned"]), f4(v["ideal"]), f4(v["learned"] - v["ideal"]) if v["ideal"] is not None else "n/a",
             v["anchors1"], pc(v["purity1"]), v["anchors2"], pc(v["purity2"]), v["min_agree_used"]]
            for c, v in d["columns"].items()]
    return table(["Column", "Uniqueness", "Learned reliability", "Ideal reliability", "Learned - ideal", "Anchors, min_agree 1",
                  "Purity, 1", "Anchors, min_agree 2", "Purity, 2", "Level used"], rows)


def cand_rows(dsets: dict, name: str) -> list:
    d = dsets[name]
    return [d["n_candidates"], d["cand_true_dups"], pc(d["cand_true_dups"] / d["n_candidates"] if d["n_candidates"] else None)]


FINDINGS = """- **Copy H2: Verity keeps all four pooled noise columns** (uniqueness 0.54 to 0.55, so not excluded as keys) but gives them almost no weight: 0.002241 in total, from reliability 0 to 0.0099. They do cost something elsewhere: with uniqueness above 0.5 they become candidate sources and add 10,831 candidate pairs, none true duplicates, so the true-duplicate share of candidates falls from 31.1% (Copy H) to 12.5%.
- **Best cutoff**: learned weights make 0 errors on original customers, 4 on Copy H and 3 on Copy H2 (pairs, hard negatives). Equal, uniqueness-only and v1 weights make 177 to 232 on Copy H and 217 to 388 on Copy H2. Original customers is 0 for all four.
- **Learned vs ideal reliability (new code)**: largest gap on any column is 0.0007 in customers, 0.0192 in Copy H (phone) and 0.0396 in Copy C (email). The old code overshoots ideal by 0.13 to 0.28 on email, phone, city and zip_code in Copy H and customers (and by 0.32 and 0.47 on email and phone in Copy C, where it had no candidates and fell back to 1.0), because it has no chance correction.
- **Ideal depends on the comparator**: email and phone in Copy C and Copy H are DESCRIPTIVE with fuzzy comparison under the old uniqueness (ideal 0.53 to 0.69) and IDENTIFIER with exact comparison under the new (ideal 0.15 to 0.21). The new uniqueness pushed them over 0.90.
- **Anchor purity**: at min_agree 2 every column is 98.5% to 100% pure in every dataset. At min_agree 1 purity is 25% to 52% in Copy H (the default run never uses it because every column has at least 270 anchors at level 2), 83% to 84% in customers (city 98%) and 89% to 99.7% in Copy C.
- **Candidates**: Copy C goes from 0 to 1675 candidates, 91.5% true duplicates. Copy H 30.8% to 31.1%. customers unchanged at 82.9%.
- **Copy C email/phone**: see section 5. Chance correction was a no-op because they became exact-match columns whose random-pair mean is 0.0."""

TRACE = """Copy C email and phone reliability are 0.123077 and 0.227723 both before and after change 2.

Path (new code):
1. Change 1 (`column_uniqueness_and_nulls`, `rule_learner.py:164`) divides distinct values by non-null count, so Copy C email and phone uniqueness rise from 0.4405 / 0.4125 to 0.9201 / 0.9348. Both are above 0.90, so `infer_semantic_type` (`rule_learner.py:221-222` for email, `:241` for phone) returns IDENTIFIER and `field_similarity` (`rule_learner.py:421`) returns 1.0 for equal values and 0.0 otherwise (`comparison_method` = exact).
2. `learn_rules` calls `learn_reliability_v2(..., random_pairs=pairs)` (`core.py:233`). Inside it, the random-pair block (`rule_learner.py:721-735`) averages `field_similarity` over the 1000 random pairs where both values are non-null: 217 pairs for email and 194 for phone. None of them are equal, so `random_mean` is exactly 0.0 for both.
3. The correction at `rule_learner.py:743`, `(anchor_mean - r) / (1.0 - r)` with r = 0.0, equals anchor_mean. The ceiling branch (`rule_learner.py:738`, r >= 0.999) and the all-zero fallback were not taken, and there are no relaxation notes.
So reliability equals the raw anchor mean (130 anchors for email, 101 for phone, min_agree 2): 0.123077 = 16/130 and 0.227723 = 23/101.

Checked by running `learn_reliability_v2` on the same candidates with `random_pairs=None` (correction off) and with the 1000 random pairs: identical output for both columns. Every other column has a nonzero random mean (0.17 to 0.51) and moves by 0.004 to 0.04."""


def h2_section(d) -> str:
    """Copy H2: columns kept with uniqueness, null rate, reliability and weight; what was excluded; candidates."""
    rows = [[c, v["semantic_type"], v["comparison_method"], f"{v['uniqueness']:.4f}", f"{v['null_rate']:.4f}",
             f"{v['learned']:.4f}", f"{v['ideal']:.4f}", f"{v['weight']:.6f}"] for c, v in d["columns"].items()]
    L = ["Copy H2 = Copy H settings (blank 50% of email and phone; typos in 30% of first_name, last_name, address; scramble 60% "
         "of zip_code and city) with the four unique noise columns replaced by `--noise-pool-cols 2 --pool-size 3000`: two "
         "8-letter-word columns and two 10-digit columns, each drawn with replacement from its own pool of 3000 random "
         "values. Built with: `python tools/degrade.py data/customers.csv data/customers_degraded_H2.csv --blank email,phone "
         "0.5 --typo first_name,last_name,address 0.3 --scramble zip_code,city 0.6 --noise-pool-cols 2 --pool-size 3000`. "
         "Truth file unchanged.", "",
         "Excluded: " + "; ".join(f"{e['name']} ({e['rule'] if isinstance(e['rule'], str) else '+'.join(e['rule'])})"
                                  for e in d["excluded"]) + ". Verity **keeps all four noise columns** (they are not unique "
         "enough to look like keys).", ""]
    L += table(["Column", "Semantic type", "Comparison", "Uniqueness", "Null rate", "Learned reliability",
                "Ideal reliability", "Composite weight"], rows)
    noise = [c for c in d["columns"] if c.startswith("noise_")]
    L += [f"Weight on the four noise columns: {sum(d['columns'][c]['weight'] for c in noise):.6f} in total. "
          f"Separability gap: {d['separability']['gap']}.", "",
          f"Candidate pairs: {d['n_candidates']}, of which {d['cand_true_dups']} are true duplicates "
          f"({pc(d['cand_true_dups'] / d['n_candidates'])}). Pairs by source column (true duplicates): "
          + ", ".join(f"{c}: {n} ({d['cand_dups_by_source'].get(c, 0)})" for c, n in d["cand_by_source"].items())
          + ". The four noise columns (uniqueness above 0.5) became candidate sources and contribute "
          f"{sum(n for c, n in d['cand_by_source'].items() if c.startswith('noise_'))} pairs, none of them true duplicates.", ""]
    return "\n".join(L)


def report(new, old, cutoff_md, h2, out):
    L = ["# Reliability diagnosis 2", "",
         "Report only. Nothing in `verity_rules/` was changed. Seed 42. New = current code (HEAD 5712630). Old = commit b2436de "
         "(old uniqueness: distinct values over all rows, raw anchor-mean reliability), imported from a git worktree.", "",
         "Definitions. **Learned reliability**: `reliability_score` from `learn_rules` of that code. **Ideal reliability**: "
         "`clip((dup mean - random mean) / (1 - random mean), 0, 1)` where dup mean is the mean `field_similarity` over every true "
         "duplicate pair with both values non-null, and random mean is the package's own random-pair mean (the 1000 random pairs "
         "`learn_rules` samples). **Anchor purity**: share of that column's anchor pairs that are true duplicates (anchors with "
         "relax off, so each min_agree level shows its own). With the old code the learned value is the raw anchor mean (no "
         "chance correction existed), so it is compared against the same chance-corrected ideal and is not expected to match it.", ""]
    L += ["## Findings", "", FINDINGS, "", "## 1. Copy H2", "", h2_section(h2), ""]
    L += ["## 2. Best cutoff", "", cutoff_md, ""]
    L += ["## 3. Learned vs ideal reliability, anchors and purity", ""]
    for label, src in (("New code (HEAD)", new), ("Old code (b2436de)", old)):
        L += [f"### {label}", ""]
        for name, _ in DATASETS:
            d = src["datasets"][name]
            L += [f"#### {name}", "",
                  f"{d['n_true_dup_pairs']} true duplicate pairs. Candidate pairs: {d['n_candidates']}. "
                  f"Kept: {', '.join(d['columns'])}."
                  + ("" if d["n_candidates"] else " No candidate pairs, so no anchors; every column's learned reliability is the "
                     "fallback 1.0."), ""]
            L += rel_table(d)
    L += ["## 4. Candidate pairs: share that are true duplicates", ""]
    rows = []
    for name, _ in DATASETS:
        o, n_ = old["datasets"][name], new["datasets"][name]
        rows.append([name, *cand_rows(old["datasets"], name), *cand_rows(new["datasets"], name)])
    L += table(["Dataset", "Old: candidate pairs", "Old: true duplicates", "Old: share", "New: candidate pairs",
                "New: true duplicates", "New: share"], rows)
    L += ["Candidate source columns (uniqueness >= 0.5). Pairs counted under the column whose group produced them first "
          "(first column in profile order wins when a pair shares values in several). Format: pairs (true duplicates).", ""]
    rows = []
    for name, _ in DATASETS:
        for label, src in (("old", old), ("new", new)):
            d = src["datasets"][name]
            allc = list(d["columns"])
            rows.append([name, label, ", ".join(f"{c} ({d['uniqueness_all'][c]:.3f})" for c in d["cand_source_columns"]) or "none",
                         ", ".join(f"{c}: {d['cand_by_source'].get(c, 0)} ({d['cand_dups_by_source'].get(c, 0)})"
                                   for c in d["cand_source_columns"] if c in d["cand_by_source"]) or "none"])
    L += table(["Dataset", "Code", "Columns qualifying as sources (uniqueness)", "Pairs by source"], rows)
    L += ["## 5. Copy C: why email and phone reliability do not change with change 2", "", TRACE, ""]
    Path(out).write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--code-root", type=Path, required=True)
    d.add_argument("--label", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--only", default=None, help="comma-separated dataset names to restrict to")
    r = sub.add_parser("report")
    r.add_argument("--new", type=Path, required=True)
    r.add_argument("--old", type=Path, required=True)
    r.add_argument("--cutoff", type=Path, required=True)
    r.add_argument("--h2", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "dump":
        ds = [x for x in DATASETS if a.only is None or x[0] in a.only.split(",")]
        a.out.write_text(json.dumps(dump(a.code_root.resolve(), a.label, ds), indent=1), encoding="utf-8")
    else:
        report(json.loads(a.new.read_text("utf-8")), json.loads(a.old.read_text("utf-8")),
               a.cutoff.read_text("utf-8"), json.loads(a.h2.read_text("utf-8")), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
