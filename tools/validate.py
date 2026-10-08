"""
Validate Verity's rules against ground truth. Not shipped to users.
This is the only place that reads truth files. It changes nothing in verity_rules/.

  .venv\\Scripts\\python.exe tools/validate.py            # writes validation/step7_results.md

Sections: stability across seeds, calibration pair purity, do the weights help (ROC AUC),
products anchor purity, degraded copies of customers (run through the CLI), and a
per-column comparator baseline. Everything is seeded, so two runs give identical output
apart from the date line.
"""
from __future__ import annotations

import argparse
import itertools
import random
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verity_rules.cli import load_records  # noqa: E402
from verity_rules.column_classification import classify_columns  # noqa: E402
from verity_rules.core import _build_candidates, learn_rules  # noqa: E402
from verity_rules.rule_learner import (  # noqa: E402
    build_profiles, field_similarity, is_null, learn_reliability_v2,
    normalize_discriminative_weights, numeric_series_stats, record_id_from_row,
    weighted_pair_similarity,
)
from degrade import degrade  # noqa: E402

SEED = 42
SEEDS = [1, 2, 3, 4, 5]
WEIGHT_RANGE_LIMIT = 0.05
TARGET_MATCH, TARGET_NON_MATCH = 0.90, 0.95
N_RANDOM_NEG, HARD_NEG_CAP = 5000, 5000
UNIQ_TOLERANCE = 0.005
SCHEMES = ["learned", "equal", "v1", "uniqueness"]


class Dataset:
    """Records, truth and a rebuilt scoring context (profiles, numeric ranges) for one CSV."""

    def __init__(self, name: str, csv: Path, truth_csv: Path, columns: list[str] | None = None,
                 variant: str = "v4"):
        self.name, self.csv, self.variant = name, csv, variant
        self.records, _ = load_records(csv, 20_000, SEED)
        self.result = learn_rules(self.records, seed=SEED, debug=True, comparator_variant=variant)
        self.columns = columns if columns is not None else list(self.result["columns"])
        df = pd.DataFrame(self.records)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.where(pd.notna(df), None)
        cls = classify_columns(df)
        self.rows = df.to_dict(orient="records")
        content = [h for h in df.columns if h in set(self.columns)]
        self.profiles = build_profiles(content, self.rows, set(cls["metadata_columns"]))
        normalize_discriminative_weights(self.profiles)
        self.lo_hi = {p.name: numeric_series_stats(p.values) for p in self.profiles}
        rid = self.result["record_id_column"]
        self.row_ids = [record_id_from_row(r, rid) for r in self.rows]
        t = pd.read_csv(truth_csv, dtype=str, keep_default_na=False)
        tmap = dict(zip(t.iloc[:, 0], t["true_entity_id"]))
        self.entity = [tmap[r] for r in self.row_ids]
        self.index = {r: i for i, r in enumerate(self.row_ids)}
        groups: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(self.entity):
            groups[e].append(i)
        self.dup_pairs = sorted(p for g in groups.values() if len(g) > 1
                                for p in itertools.combinations(g, 2))

    def is_dup(self, i: int, j: int) -> bool:
        return self.entity[i] == self.entity[j]

    def set_weights(self, weights: dict[str, float]) -> None:
        for p in self.profiles:
            p.composite_weight = weights[p.name]

    def score(self, pairs: list[tuple[int, int]]) -> list[float]:
        return [weighted_pair_similarity(self.rows[i], self.rows[j], self.profiles, self.lo_hi,
                                         weight_attr="composite_weight",
                                         comparator_variant=self.variant) for i, j in pairs]

    def random_non_dup(self, k: int, seed: int = SEED) -> list[tuple[int, int]]:
        rng, n, out = random.Random(seed), len(self.rows), set()
        while len(out) < k:
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j and not self.is_dup(i, j):
                out.add((min(i, j), max(i, j)))
        return sorted(out)

    def hard_negatives(self, seed: int = SEED) -> tuple[list[tuple[int, int]], int, int]:
        cands = sorted(_build_candidates(self.rows, self.profiles))
        neg = [p for p in cands if not self.is_dup(*p)]
        n_all = len(neg)
        if len(neg) > HARD_NEG_CAP:
            neg = sorted(random.Random(seed).sample(neg, HARD_NEG_CAP))
        return neg, len(cands), n_all

    def weight_schemes(self, learned: dict[str, float], v1: dict[str, float] | None) -> dict[str, dict[str, float]]:
        names = [p.name for p in self.profiles]
        u = {p.name: p.uniqueness_ratio for p in self.profiles}
        su = sum(u.values())
        out = {"learned": learned, "equal": {c: 1 / len(names) for c in names},
               "uniqueness": {c: u[c] / su for c in names}}
        if v1 is not None:
            out["v1"] = v1
        return out


def auc(ds: Dataset, weights: dict[str, float], neg: list[tuple[int, int]]) -> float:
    ds.set_weights(weights)
    pos = ds.score(ds.dup_pairs)
    ng = ds.score(neg)
    if not pos or not ng:
        return float("nan")
    return float(roc_auc_score([1] * len(pos) + [0] * len(ng), pos + ng))


def best_cutoff(pos: list[float], neg: list[float]) -> dict[str, int]:
    """Scan every cutoff (a pair is merged when its score >= cutoff, over every distinct score plus one
    above the maximum) and return the one with the fewest errors: missed duplicates (positives below the
    cutoff) plus false merges (negatives at or above it). Counts of pairs. Ties go to the higher cutoff."""
    p, g = np.sort(np.asarray(pos, dtype=float)), np.sort(np.asarray(neg, dtype=float))
    cuts = np.unique(np.concatenate([p, g, [np.inf]]))
    missed = np.searchsorted(p, cuts, side="left")
    false = len(g) - np.searchsorted(g, cuts, side="left")
    k = int(np.argmin((missed + false)[::-1]))  # reversed, so argmin returns the highest cutoff on ties
    k = len(cuts) - 1 - k
    return {"errors": int(missed[k] + false[k]), "missed": int(missed[k]), "false_merges": int(false[k]),
            "n_pos": len(p), "n_neg": len(g)}


def cutoff_rows(ds: "Dataset", schemes: dict[str, dict[str, float]], neg_hard: list[tuple[int, int]],
                neg_rand: list[tuple[int, int]]) -> list[list[Any]]:
    """One row per weighting: AUC and best-cutoff errors against hard and random negatives."""
    rows = []
    for name, w in schemes.items():
        ds.set_weights(w)
        pos = ds.score(ds.dup_pairs)
        cells = [name]
        for neg in (neg_hard, neg_rand):
            ng = ds.score(neg)
            a = float(roc_auc_score([1] * len(pos) + [0] * len(ng), pos + ng)) if pos and ng else float("nan")
            b = best_cutoff(pos, ng)
            cells += [f"{a:.6f}", b["errors"], b["missed"], b["false_merges"]]
        rows.append(cells)
    return rows


CUTOFF_HEADER = ["Weighting", "AUC (hard)", "Best-cutoff errors (hard)", "missed (hard)", "false merges (hard)",
                 "AUC (random)", "Best-cutoff errors (random)", "missed (random)", "false merges (random)"]


def best_cutoff_section(ds: "Dataset") -> list[str]:
    """Best-cutoff report for learned, equal, uniqueness-only and v1 weights on one dataset."""
    v1 = {c: v["composite_weight"] for c, v in learn_rules(ds.records, seed=SEED, reliability="v1")["columns"].items()}
    learned = {c: ds.result["columns"][c]["composite_weight"] for c in ds.columns}
    schemes = ds.weight_schemes(learned, v1)
    schemes = {k: schemes[k] for k in SCHEMES}
    neg_hard, n_cand, n_all = ds.hard_negatives()
    neg_rand = ds.random_non_dup(N_RANDOM_NEG)
    rows = cutoff_rows(ds, schemes, neg_hard, neg_rand)
    L = [f"### Best cutoff, {ds.name}", "",
         f"Positives: {len(ds.dup_pairs)} true duplicate pairs. Hard negatives: {len(neg_hard)} candidate pairs "
         f"that are not true duplicates (of {n_cand} candidates, {n_all} non-duplicate"
         + (f"; capped at {HARD_NEG_CAP}" if n_all > HARD_NEG_CAP else "") + f"). Random negatives: {len(neg_rand)}. "
         "Errors = missed duplicates + false merges at the best cutoff, in pairs.", ""]
    return L + table(CUTOFF_HEADER, rows)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def purity(ds: Dataset, result: dict[str, Any]) -> dict[str, tuple[int, int | None]]:
    """Per category: (n pairs, count that are true duplicates)."""
    out = {}
    for cat, pairs in result["calibration_pairs"].items():
        dups = sum(ds.is_dup(ds.index[p["record_id_a"]], ds.index[p["record_id_b"]]) for p in pairs)
        out[cat] = (len(pairs), dups)
    return out


# ---------------------------------------------------------------- sections

def stability(ds: Dataset, checks: list) -> list[str]:
    runs = {s: learn_rules(ds.records, seed=s)["columns"] for s in SEEDS}
    rows, flagged = [], []
    for c in ds.columns:
        ws = [runs[s][c]["composite_weight"] for s in SEEDS]
        spread = max(ws) - min(ws)
        if spread > WEIGHT_RANGE_LIMIT:
            flagged.append(c)
        rows.append([c, f"{min(ws):.6f}", f"{max(ws):.6f}", f"{spread:.6f}", "FLAG" if spread > WEIGHT_RANGE_LIMIT else ""])
    rankings = {s: tuple(sorted(runs[s], key=lambda c: (-runs[s][c]["composite_weight"], c))) for s in SEEDS}
    same = len(set(rankings.values())) == 1
    checks.append((f"{ds.name}: no composite_weight range above {WEIGHT_RANGE_LIMIT}", not flagged,
                   "flagged: " + ", ".join(flagged) if flagged else "max range "
                   + f"{max(float(r[3]) for r in rows):.6f}"))
    L = [f"### Stability, {ds.name}, seeds {SEEDS[0]} to {SEEDS[-1]}", ""]
    L += table(["Column", "Min weight", "Max weight", "Range", "Flag"], rows)
    L.append(f"Column ranking the same across seeds: **{'yes' if same else 'no'}**.")
    if same:
        L.append(f"Ranking: {' > '.join(rankings[SEEDS[0]])}.")
    else:
        L += [f"- seed {s}: {' > '.join(r)}" for s, r in rankings.items()]
    return L + [""]


def calibration(ds: Dataset, result: dict[str, Any], checks: list | None, title: str) -> list[str]:
    pur = purity(ds, result)

    def share(cat: str, want_dup: bool) -> float | None:
        n, d = pur[cat]
        return None if n == 0 else (d if want_dup else n - d) / n

    lm, nm, un = share("likely_match", True), share("likely_non_match", False), share("unclear", True)
    if checks is not None:
        checks.append((f"{ds.name}: likely_match purity >= {TARGET_MATCH:.0%}", lm is not None and lm >= TARGET_MATCH,
                       f"{pct(lm)} ({pur['likely_match'][1]} of {pur['likely_match'][0]})"))
        checks.append((f"{ds.name}: likely_non_match purity >= {TARGET_NON_MATCH:.0%}",
                       nm is not None and nm >= TARGET_NON_MATCH,
                       f"{pct(nm)} ({pur['likely_non_match'][0] - pur['likely_non_match'][1]} of {pur['likely_non_match'][0]})"))
    sep = result["separability"]
    L = [f"### {title}", "",
         f"Separability: label **{sep['label']}**, gap {sep['gap']}, "
         f"top-decile mean {sep['candidate_top_decile_mean']}, random-pair mean {sep['random_pair_mean']}.", ""]
    L += table(["Category", "Pairs", "True duplicates", "Share"], [
        ["likely_match (share true duplicates, target >= 90%)", pur["likely_match"][0], pur["likely_match"][1], pct(lm)],
        ["likely_non_match (share true non-duplicates, target >= 95%)", pur["likely_non_match"][0],
         pur["likely_non_match"][0] - pur["likely_non_match"][1], pct(nm)],
        ["unclear (share true duplicates, report only)", pur["unclear"][0], pur["unclear"][1], pct(un)],
    ])
    return L


def helps(ds: Dataset, checks: list, v1: dict[str, float]) -> list[str]:
    learned = {c: ds.result["columns"][c]["composite_weight"] for c in ds.columns}
    schemes = ds.weight_schemes(learned, v1)
    neg_rand = ds.random_non_dup(N_RANDOM_NEG)
    neg_hard, n_cand, n_hard_all = ds.hard_negatives()
    aucs = {s: (auc(ds, schemes[s], neg_rand), auc(ds, schemes[s], neg_hard)) for s in SCHEMES}
    ds.hard_count = len(neg_hard)
    L = [f"### Do the weights help, {ds.name}", "",
         f"Positives: {len(ds.dup_pairs)} true duplicate pairs. Random negatives: {len(neg_rand)}. "
         f"Hard negatives: {len(neg_hard)} (candidate pairs: {n_cand}, of which {n_hard_all} not true duplicates"
         + (f"; capped at {HARD_NEG_CAP}" if n_hard_all > HARD_NEG_CAP else "") + ").", ""]
    L += table(["Weighting", "AUC vs random negatives", "AUC vs hard negatives"],
               [[s, f"{aucs[s][0]:.6f}", f"{aucs[s][1]:.6f}"] for s in SCHEMES])
    L += ["Weights used:", ""]
    L += table(["Column"] + SCHEMES, [[c] + [f"{schemes[s][c]:.6f}" for s in SCHEMES] for c in ds.columns])
    le, lu = aucs["learned"][1], aucs["uniqueness"][1]
    undefined = np.isnan(le) or np.isnan(lu) or np.isnan(aucs["equal"][1])  # no hard negatives
    note = f" (no hard negatives: all {n_cand} candidate pairs are true duplicates)" if undefined else ""
    checks.append((f"{ds.name}: learned hard AUC >= equal hard AUC", None if undefined else le >= aucs["equal"][1],
                   f"{le:.6f} vs {aucs['equal'][1]:.6f}{note}"))
    checks.append((f"{ds.name}: learned hard AUC >= uniqueness hard AUC - {UNIQ_TOLERANCE}",
                   None if undefined else le >= lu - UNIQ_TOLERANCE,
                   f"{le:.6f} vs {lu:.6f} - {UNIQ_TOLERANCE}{note}"))
    ds.neg_rand = neg_rand
    return L


def anchor_purity(ds: Dataset) -> list[str]:
    cands = sorted(_build_candidates(ds.rows, ds.profiles))
    runs = {m: learn_reliability_v2(cands, ds.rows, ds.profiles, ds.lo_hi, min_agree=m, relax=False)[1]
            for m in (1, 2)}
    used = ds.result["reliability_debug"]["min_agree_used"]
    rows = []
    for c in ds.columns:
        row = [c]
        for m in (1, 2):
            ap = runs[m]["anchor_pairs"][c]
            row += [len(ap), pct(sum(ds.is_dup(i, j) for i, j in ap) / len(ap)) if ap else "n/a"]
        row.append(used[c])
        rows.append(row)
    L = [f"### Anchor purity, {ds.name} (relax off, so each level shows its own anchors)", "",
         "Purity is the share of anchor pairs that are true duplicates. The last column is the level the default run used.", ""]
    return L + table(["Column", "Anchors, min_agree 1", "Purity, 1", "Anchors, min_agree 2", "Purity, 2", "Level used"], rows)


def degraded(base_csv: Path, truth_csv: Path, checks: list) -> list[str]:
    id_col = pd.read_csv(truth_csv, nrows=0).columns[0]
    df = pd.read_csv(base_csv, dtype=str, keep_default_na=False)
    ec = "email,phone"
    specs = {
        "A": ("blank 25% of email, phone", dict(blank=[(ec, 0.25)], typos=[])),
        "B": ("blank 50% of email, phone; typos in 30% of first_name, last_name, address",
              dict(blank=[(ec, 0.5)], typos=[("first_name,last_name,address", 0.3)])),
        "C": ("blank 50% of every column except the record id", dict(blank=[("ALL", 0.5)], typos=[])),
    }
    rows, notes = [], []
    for key, (desc, kw) in specs.items():
        csv = ROOT / "data" / f"customers_degraded_{key}.csv"
        out_dir = ROOT / "out" / f"degraded_{key}"
        degrade(df, SEED, kw["blank"], kw["typos"], {id_col}).to_csv(csv, index=False)
        subprocess.run([sys.executable, "-m", "verity_rules", "learn", str(csv), "--out", str(out_dir)],
                       check=True, capture_output=True, text=True, cwd=ROOT)
        import json
        res = json.loads((out_dir / "rules.json").read_text(encoding="utf-8"))
        ds = Dataset(f"degraded {key}", csv, truth_csv, columns=list(res["columns"]))
        pur = purity(ds, res)
        learned = {c: res["columns"][c]["composite_weight"] for c in ds.columns}
        sch = ds.weight_schemes(learned, None)
        neg_hard, _, _ = ds.hard_negatives()
        a_l, a_e = auc(ds, sch["learned"], neg_hard), auc(ds, sch["equal"], neg_hard)
        neg_rand = ds.random_non_dup(N_RANDOM_NEG)
        r_l, r_e = auc(ds, sch["learned"], neg_rand), auc(ds, sch["equal"], neg_rand)
        lm_n, lm_d = pur["likely_match"]
        un_n, un_d = pur["unclear"]
        sep = res["separability"]
        rows.append([key, sep["gap"], sep["label"], f"{pct(lm_d / lm_n if lm_n else None)} ({lm_d}/{lm_n})",
                     f"{pct(un_d / un_n if un_n else None)} ({un_d}/{un_n})", f"{a_l:.6f}", f"{a_e:.6f}",
                     len(neg_hard), f"{r_l:.6f}", f"{r_e:.6f}", res["run"]["n_candidate_pairs"]])
        notes.append(f"- {key}: {desc}. Columns kept: {', '.join(res['columns'])}. "
                     f"Excluded: {', '.join(e['name'] for e in res['columns_excluded']) or 'none'}.")
    L = ["## Degraded copies of customers (report only, no cutoffs changed)", "",
         "Truth file unchanged. Each copy was run through the CLI (seed 42). Blank fractions are of all rows; "
         "typo fractions are of the non-empty values left after blanking. Hard negatives are built from the "
         "degraded copy. nan means the set is empty. Random-negative AUC is added so a copy with no candidates "
         "still has a number.", ""]
    L += table(["Copy", "Separability gap", "Label", "likely_match true-duplicate share", "unclear true-duplicate share",
                "AUC learned (hard)", "AUC equal (hard)", "Hard negatives",
                "AUC learned (random)", "AUC equal (random)", "Candidate pairs"], rows)
    return L + notes + [""]


def baseline(datasets: list[Dataset]) -> list[str]:
    out = []
    for ds in datasets:
        for p in ds.profiles:
            def mean_sim(pairs):
                vals = []
                for i, j in pairs:
                    va, vb = ds.rows[i].get(p.name), ds.rows[j].get(p.name)
                    if is_null(va) or is_null(vb):
                        continue
                    vals.append(field_similarity(va, vb, p, ds.lo_hi[p.name]))
                return (float(np.mean(vals)) if vals else None), len(vals)
            m_rand, n_rand = mean_sim(ds.neg_rand)
            m_dup, n_dup = mean_sim(ds.dup_pairs)
            diff = None if m_rand is None or m_dup is None else m_dup - m_rand
            out.append((diff, ds.name, p.name, ds.result["columns"][p.name]["comparison_method"],
                        m_rand, n_rand, m_dup, n_dup))
    out.sort(key=lambda r: (r[0] is None, r[0], r[1], r[2]))
    f = lambda x: "n/a" if x is None else f"{x:.6f}"  # noqa: E731
    L = ["## Comparator baseline", "",
         "Mean field_similarity with each column's own comparison method, over pairs where both values are non-null. "
         f"Non-duplicate set: the same {N_RANDOM_NEG} random non-duplicate pairs used above. Duplicate set: all true "
         "duplicate pairs. Difference = duplicate mean minus non-duplicate mean, smallest first.", ""]
    L += table(["Dataset", "Column", "Method", "Non-dup mean", "n", "Dup mean", "n", "Difference"],
               [[d, c, m, f(mr), nr, f(md), nd, f(df)] for df, d, c, m, mr, nr, md, nd in out])
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "validation" / "step7_results.md")
    ap.add_argument("--best-cutoff", nargs="+", metavar="NAME=CSV", default=None,
                    help="only write the best-cutoff section for these datasets (name=path to a customers-style CSV, "
                         "truth from data/customers_truth.csv) to --out, then exit")
    args = ap.parse_args(argv)
    data = ROOT / "data"
    if args.best_cutoff:
        L = ["# Best cutoff", ""]
        for spec in args.best_cutoff:
            name, csv = spec.split("=", 1)
            L += best_cutoff_section(Dataset(name, Path(csv), data / "customers_truth.csv"))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(L), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    checks: list[tuple[str, bool | None, str]] = []
    body: list[str] = []
    datasets = []
    for name in ("customers", "products"):
        ds = Dataset(name, data / f"{name}.csv", data / f"{name}_truth.csv")
        datasets.append(ds)
        print(f"loaded {name}: {len(ds.rows)} rows, {len(ds.dup_pairs)} duplicate pairs", flush=True)

    body += ["## Stability across seeds", ""]
    for ds in datasets:
        body += stability(ds, checks)
    body += ["## Calibration pairs against truth (seed 42)", ""]
    for ds in datasets:
        body += calibration(ds, ds.result, checks, f"Calibration, {ds.name}")
    body += ["## Do the weights help?", "",
             "Pair scoring is the package's weighted_pair_similarity with the weights swapped. learned = published "
             "composite_weight (v2). equal = 1 / number of content columns. v1 = composite_weight from "
             "learn_rules(reliability='v1'), seed 42. uniqueness = uniqueness_ratio normalised to sum 1, no "
             "reliability and no null-rate factor.", ""]
    for ds in datasets:
        v1 = {c: v["composite_weight"] for c, v in learn_rules(ds.records, seed=SEED, reliability="v1")["columns"].items()}
        body += helps(ds, checks, v1)
    body += ["## Products anchor purity", ""]
    body += anchor_purity(datasets[1])
    body += degraded(data / "customers.csv", data / "customers_truth.csv", checks)
    body += baseline(datasets)

    passed = all(ok is True for _, ok, _ in checks)
    n_pass = sum(ok is True for _, ok, _ in checks)
    n_fail = sum(ok is False for _, ok, _ in checks)
    n_na = sum(ok is None for _, ok, _ in checks)
    label = lambda ok: "PASS" if ok else "NOT EVALUABLE" if ok is None else "FAIL"  # noqa: E731
    top = ["# Step 7 validation results", "", f"Date: {date.today().isoformat()}", "",
           f"**Overall: {'PASS' if passed else 'FAIL' if n_fail else 'NOT FULLY EVALUATED'}** "
           f"({n_pass} of {len(checks)} targets met, {n_fail} failed, {n_na} not evaluable). "
           "Nothing was tuned. Degraded copies, anchor purity and the comparator baseline are report only.", ""]
    top += table(["Target", "Result", "Value"], [[n, label(ok), d] for n, ok, d in checks])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(top + body), encoding="utf-8")
    print(f"wrote {args.out}: {n_pass} pass, {n_fail} fail, {n_na} not evaluable")
    for n, ok, d in checks:
        print(label(ok) + " " + n + " :: " + d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
