"""
Diagnose why a column's reliability_score is what it is.

For each content column, reports:
  1. The raw reliability (support / (support + conflict)) before the 0.1-1.0
     rescale, and how many anchor pairs learn_reliability used. learn_reliability
     does not expose the raw value, so this script repeats its loop using the
     package helpers (same profiles, same pair sample, same seed). The package
     is imported, not edited.
  2. Across all true duplicate pairs from the truth file: share with both values
     non-null, share exactly equal after lowercasing and trimming, and mean
     char_jaccard. Then three example true-duplicate pairs for the two columns
     with the lowest exact agreement.

Not shipped to users. Reads a truth file, so it lives in tools/.

Usage:
  python tools/diagnose_reliability.py data/customers.csv data/customers_truth.csv customer_id
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verity_rules.column_classification import classify_columns  # noqa: E402
from verity_rules.rule_learner import (  # noqa: E402
    build_profiles, char_jaccard, clean_str, exact_match, is_null,
    normalize_discriminative_weights, numeric_series_stats,
)

TRUTH_COLUMN = "true_entity_id"
PAIR_SAMPLE = 1000  # learn_rules default
N_EXAMPLES = 3


def sample_pairs(n: int, pair_sample: int, seed: int) -> list[tuple[int, int]]:
    """Same pair sampling as learn_rules."""
    random.seed(seed)
    np.random.seed(seed)
    if n * (n - 1) // 2 <= pair_sample:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    s: set[tuple[int, int]] = set()
    while len(s) < pair_sample:
        i, j = random.randrange(n), random.randrange(n)
        if i != j:
            s.add((min(i, j), max(i, j)))
    return list(s)


def raw_reliability(pairs, rows, profiles, lo_hi):
    """Repeat learn_reliability up to the rescale. Returns (raw, anchors_used, per-column anchors)."""
    support: dict[str, float] = defaultdict(float)
    conflict: dict[str, float] = defaultdict(float)
    col_anchor_pairs: dict[str, int] = defaultdict(int)
    w_by_col = {p.name: max(p.normalized_weight, 1e-6) for p in profiles}
    n_anchor = 0
    for i, j in pairs:
        ri, rj = rows[i], rows[j]
        matched = [p for p in profiles
                   if exact_match(ri.get(p.name), rj.get(p.name), p, lo_hi[p.name])]
        if not matched:
            continue
        anchor_w = sum(w_by_col[m.name] for m in matched)
        if anchor_w <= 0:
            continue
        n_anchor += 1
        for p in profiles:
            vi, vj = ri.get(p.name), rj.get(p.name)
            if is_null(vi) or is_null(vj):
                continue
            col_anchor_pairs[p.name] += 1
            if exact_match(vi, vj, p, lo_hi[p.name]):
                support[p.name] += anchor_w
            else:
                conflict[p.name] += anchor_w
    raw = {p.name: support.get(p.name, 0.0) / (support.get(p.name, 0.0) + conflict.get(p.name, 0.0) + 1e-9)
           for p in profiles}
    return raw, n_anchor, col_anchor_pairs


def true_duplicate_pairs(ids: list[str], truth: dict[str, str]) -> list[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, rid in enumerate(ids):
        if rid in truth:
            groups[truth[rid]].append(idx)
    return [pair for members in groups.values() for pair in combinations(members, 2)]


def diagnose(data_path: Path, truth_path: Path, id_col: str, seed: int) -> None:
    df = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    tdf = pd.read_csv(truth_path, dtype=str, keep_default_na=False)
    truth = dict(zip(tdf[id_col], tdf[TRUTH_COLUMN]))

    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)
    cls = classify_columns(df)
    rows = df.to_dict(orient="records")
    content = [h for h in df.columns if h in set(cls["content_columns"])]
    profiles = build_profiles(content, rows, set(cls["metadata_columns"]))
    normalize_discriminative_weights(profiles)
    lo_hi = {p.name: numeric_series_stats(p.values) for p in profiles}

    pairs = sample_pairs(len(rows), PAIR_SAMPLE, seed)
    raw, n_anchor, col_anchor = raw_reliability(pairs, rows, profiles, lo_hi)

    ids = [str(r[id_col]).strip() for r in rows]
    dups = true_duplicate_pairs(ids, truth)

    print(f"Dataset: {data_path}  records={len(rows)}  id column={id_col}  seed={seed}")
    print(f"Sampled pairs: {len(pairs)}  anchor pairs used by learn_reliability: {n_anchor}")
    print(f"True duplicate pairs: {len(dups)}")
    print(f"Content columns: {content}\n")

    stats: dict[str, dict] = {}
    for p in profiles:
        both = equal = 0
        jac: list[float] = []
        for i, j in dups:
            vi, vj = rows[i].get(p.name), rows[j].get(p.name)
            if is_null(vi) or is_null(vj):
                continue
            both += 1
            a, b = clean_str(vi).lower(), clean_str(vj).lower()
            equal += a == b
            jac.append(char_jaccard(a, b))
        d = max(len(dups), 1)
        stats[p.name] = {
            "both": both / d,
            "equal": equal / d,
            "equal_given_both": equal / both if both else 0.0,
            "jaccard": float(np.mean(jac)) if jac else 0.0,
        }

    hdr = (f"{'column':<26}{'type':<18}{'raw_rel':>8}{'col_anchors':>12}"
           f"{'both_nonnull':>13}{'exact_eq':>9}{'eq|both':>8}{'mean_jacc':>10}")
    print(hdr)
    print("-" * len(hdr))
    for p in profiles:
        s = stats[p.name]
        print(f"{p.name:<26}{p.semantic_type:<18}{raw[p.name]:>8.4f}{col_anchor[p.name]:>12}"
              f"{s['both']:>13.3f}{s['equal']:>9.3f}{s['equal_given_both']:>8.3f}{s['jaccard']:>10.3f}")
    print("\nraw_rel      = support/(support+conflict) before the 0.1-1.0 rescale")
    print("col_anchors  = anchor pairs where this column was non-null on both sides")
    print("both_nonnull / exact_eq = share of ALL true duplicate pairs; exact_eq is after lowercase+trim")
    print("eq|both      = exact_eq among pairs where both values are non-null")
    print("mean_jacc    = mean char_jaccard over pairs where both values are non-null")

    lowest = sorted(stats, key=lambda c: stats[c]["equal"])[:2]
    rng = random.Random(seed)
    for col in lowest:
        usable = [(i, j) for i, j in dups
                  if not is_null(rows[i].get(col)) and not is_null(rows[j].get(col))]
        picks = rng.sample(usable, min(N_EXAMPLES, len(usable)))
        print(f"\nExamples for '{col}' (lowest exact agreement; true duplicates with both values non-null)")
        for i, j in picks:
            print(f"  {ids[i]} / {ids[j]}  entity={truth[ids[i]]}")
            print(f"    A: {rows[i].get(col)!r}")
            print(f"    B: {rows[j].get(col)!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_csv", type=Path)
    ap.add_argument("truth_csv", type=Path)
    ap.add_argument("id_column")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    diagnose(a.data_csv, a.truth_csv, a.id_column, a.seed)


if __name__ == "__main__":
    main()
