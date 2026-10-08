"""
Pair-score formula experiment (S0 to S3). Not shipped to users. Report only; learned weights are not changed.

  .venv\\Scripts\\python.exe tools/scoring_experiment.py   # writes validation/scoring_experiment.md

Weights are the v4 composite weights from learn_rules (default comparator_variant) for each dataset. For a pair,
let present = columns with a value in both records, w = weight, s = column similarity, W = sum of all weights:
  S0 = sum(w*s over present) / sum(w over present)      current rule: skip missing columns and rescale
  S1 = sum(w*s over present) / W                        missing columns add 0 to the numerator, W in the denominator
  S2 = S0 * coverage                                    coverage = sum(w over present) / W
  S3 = S0 * sqrt(coverage)
Positives: every true duplicate pair. Negatives: hard negatives (candidate pairs that are not true duplicates,
capped at 5,000 with seed 42). Coverage buckets for Copy C and Copy C25 use coverage, not the score.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import Dataset, HARD_NEG_CAP, SEED, best_cutoff, table  # noqa: E402
from verity_rules.rule_learner import field_similarity, is_null  # noqa: E402

DATA = ROOT / "data"
SETS = [("customers", "customers.csv"), ("Copy C", "customers_degraded_C.csv"),
        ("Copy H", "customers_degraded_H.csv"), ("Copy H2", "customers_degraded_H2.csv"),
        ("Copy C25", "customers_degraded_C25.csv")]
BUCKET_SETS = {"Copy C", "Copy C25"}
FORMULAS = ["S0", "S1", "S2", "S3"]
BUCKETS = [("below 0.3", lambda c: c < 0.3), ("0.3 to 0.6", lambda c: 0.3 <= c <= 0.6),
           ("above 0.6", lambda c: c > 0.6)]


def pair_scores(ds: Dataset, pairs: list[tuple[int, int]]) -> list[dict[str, float]]:
    total_w = sum(p.composite_weight for p in ds.profiles)
    out = []
    for i, j in pairs:
        ri, rj = ds.rows[i], ds.rows[j]
        num = den = 0.0
        for p in ds.profiles:
            vi, vj = ri.get(p.name), rj.get(p.name)
            if is_null(vi) or is_null(vj):
                continue
            s = field_similarity(vi, vj, p, ds.lo_hi[p.name], comparator_variant=ds.variant)
            num += p.composite_weight * s
            den += p.composite_weight
        s0 = num / den if den > 0 else 0.0
        cov = den / total_w if total_w > 0 else 0.0
        out.append({"S0": s0, "S1": num / total_w if total_w > 0 else 0.0, "S2": s0 * cov,
                    "S3": s0 * math.sqrt(cov), "cov": cov})
    return out


def run(name: str, csv: str) -> list[str]:
    ds = Dataset(name, DATA / csv, DATA / "customers_truth.csv")
    ds.set_weights({p.name: ds.result["columns"][p.name]["composite_weight"] for p in ds.profiles})
    neg_pairs, n_cand, n_all = ds.hard_negatives()
    pos, neg = pair_scores(ds, ds.dup_pairs), pair_scores(ds, neg_pairs)
    # S0 here must equal the library's weighted_pair_similarity
    lib = ds.score(ds.dup_pairs[:200])
    assert np.allclose(lib, [d["S0"] for d in pos[:200]], atol=1e-12), "S0 differs from weighted_pair_similarity"
    L = [f"## {name}", "",
         f"Columns and weights: " + ", ".join(f"{p.name} {p.composite_weight:.3f}" for p in
                                              sorted(ds.profiles, key=lambda p: -p.composite_weight)) + ".", "",
         f"{len(pos)} true duplicate pairs; {len(neg)} hard negatives (of {n_cand} candidates, {n_all} non-duplicate"
         + (f"; capped at {HARD_NEG_CAP}" if n_all > HARD_NEG_CAP else "") + ").", ""]
    rows = []
    for f in FORMULAS:
        ps, ns = [d[f] for d in pos], [d[f] for d in neg]
        auc = float(roc_auc_score([1] * len(ps) + [0] * len(ns), ps + ns)) if ps and ns else float("nan")
        b = best_cutoff(ps, ns)
        rows.append([f, f"{auc:.6f}", b["errors"], b["missed"], b["false_merges"]])
    L += table(["Formula", "AUC (hard)", "Best-cutoff errors", "missed duplicates", "false merges"], rows)
    L += [f"Mean coverage: duplicates {np.mean([d['cov'] for d in pos]):.3f}, "
          f"hard negatives {np.mean([d['cov'] for d in neg]):.3f}.", ""]
    if name in BUCKET_SETS:
        brow = []
        for label, test in BUCKETS:
            cells = [label]
            for grp in (pos, neg):
                sel = [d["S0"] for d in grp if test(d["cov"])]
                cells += [len(sel), f"{np.mean(sel):.3f}" if sel else "n/a"]
            brow.append(cells)
        L += ["Coverage buckets (coverage = sum of weights of columns present in both records / sum of all weights). "
              "Mean score is S0.", ""]
        L += table(["Coverage", "Duplicates: count", "Duplicates: mean S0", "Hard negatives: count",
                    "Hard negatives: mean S0"], brow)
    return L


def main() -> int:
    L = ["# Pair-score formula experiment", "",
         "Branch `exp-scoring`. Seed 42. Report only; nothing in `verity_rules/` changed and the learned weights are the "
         "v4 composite weights. Generated by `tools/scoring_experiment.py`.", "",
         "Formulas. For a pair, *present* = columns with a value in both records, w = column weight, s = column similarity, "
         "W = sum of all weights.", "",
         "- **S0** (current rule): sum(w*s over present) / sum(w over present). Missing columns are skipped and the rest rescaled.",
         "- **S1**: sum(w*s over present) / W. Missing columns add 0 to the numerator; the denominator is the sum of all weights.",
         "- **S2**: S0 * coverage, where coverage = sum(w over present) / W.",
         "- **S3**: S0 * sqrt(coverage).", "",
         "Negatives are hard negatives: candidate pairs (identical value in a highly distinctive column) that are not true "
         "duplicates, capped at 5,000. Best-cutoff errors = missed duplicates + false merges at the best single cutoff, in pairs.", "",
         "Copy C = blank 50% of every column except `customer_id`; Copy C25 = blank 25% of every column except `customer_id` "
         "(`python tools/degrade.py data/customers.csv data/customers_degraded_C25.csv --seed 42 --blank ALL 0.25 "
         "--except customer_id`). Copy H and H2 are described in `validation/reliability_cc_results.md` and "
         "`validation/reliability_diagnosis_2.md`.", ""]
    for name, csv in SETS:
        print(name, flush=True)
        L += run(name, csv)
    out = ROOT / "validation" / "scoring_experiment.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
