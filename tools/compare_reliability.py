"""
Compare reliability v1 and v2 against the ideal reliability from a truth file.

Ideal reliability for a column = mean field_similarity over true duplicate pairs
where both values are non-null.

Per dataset, prints:
  - per-column reliability: ideal, v1, v2 with the 0.1-1.0 rescale, v2 without it
  - Spearman rank correlation with the ideal
  - final composite weights (normalized_weight * reliability, renormalized) for each variant
  - anchor counts per column and anchor purity against truth
  - if v2 purity is under 90 percent, the v2 report again with agreement required
    on at least 2 other columns
  - the v1 anchor pairs (sampled pairs with at least one exact-matching column)

Reads truth files, so it lives in tools/ and is not shipped.

Usage:
  python tools/compare_reliability.py \\
      --dataset data/customers.csv data/customers_truth.csv customer_id \\
      --dataset data/products.csv data/products_truth.csv product_id \\
      --out validation/reliability_v2_comparison.txt
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
from verity_rules.core import (  # noqa: E402
    CANDIDATE_CAP, _build_candidates, learn_rules,
)
from verity_rules.rule_learner import (  # noqa: E402
    build_profiles, exact_match, field_similarity, is_null, learn_reliability,
    learn_reliability_v2, normalize_discriminative_weights, numeric_series_stats,
)

TRUTH_COLUMN = "true_entity_id"
PURITY_FLOOR = 0.90
MAX_V1_ANCHORS_SHOWN = 20
SEED = 42
PAIR_SAMPLE = 1000  # learn_rules default

OUT: list[str] = []


def emit(line: str = "") -> None:
    OUT.append(line)
    print(line)


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


def composite(profiles, rel: dict[str, float]) -> dict[str, float]:
    comps = {p.name: max(1e-9, p.normalized_weight * rel[p.name]) for p in profiles}
    total = sum(comps.values())
    return {k: v / total for k, v in comps.items()}


def spearman(a: list[float], b: list[float]) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def purity(pairs, ids, truth) -> tuple[int, float]:
    """(count, share of pairs whose two records share a true entity)."""
    if not pairs:
        return 0, float("nan")
    ok = sum(1 for i, j in pairs
             if ids[i] in truth and ids[j] in truth and truth[ids[i]] == truth[ids[j]])
    return len(pairs), ok / len(pairs)


def fmt(x: float, w: int = 8, d: int = 4) -> str:
    return f"{'n/a':>{w}}" if x != x else f"{x:>{w}.{d}f}"


def v2_section(label: str, min_agree: int, ctx: dict) -> float:
    """Print a full v2 report. Returns overall anchor purity (min_agree as given)."""
    profiles, rows, lo_hi = ctx["profiles"], ctx["rows"], ctx["lo_hi"]
    ids, truth, ideal = ctx["ids"], ctx["truth"], ctx["ideal"]
    cand = ctx["cand_pairs"]
    names = [p.name for p in profiles]

    rel_rs, info = learn_reliability_v2(cand, rows, profiles, lo_hi,
                                        min_agree=min_agree, rescale=True)
    rel_raw, _ = learn_reliability_v2(cand, rows, profiles, lo_hi,
                                      min_agree=min_agree, rescale=False)
    union = sorted({pr for lst in info["anchor_pairs"].values() for pr in lst})
    n_union, pur_union = purity(union, ids, truth)

    emit(f"--- {label} ---")
    emit(f"Anchor pairs (union over columns): {n_union}  purity vs truth: "
         f"{pur_union:.1%}" if n_union else f"Anchor pairs (union over columns): 0")
    emit(f"{'column':<26}{'anchors':>8}{'purity':>8}{'ideal':>8}{'v2_rescaled':>12}{'v2_raw':>8}")
    for nm in names:
        n, pu = purity(info["anchor_pairs"][nm], ids, truth)
        emit(f"{nm:<26}{n:>8}{fmt(pu, 8, 3)}{fmt(ideal[nm])}{fmt(rel_rs[nm], 12)}{fmt(rel_raw[nm])}")
    for note in info["notes"]:
        emit(f"  note: {note}")
    iv = [ideal[nm] for nm in names]
    emit(f"Spearman vs ideal:  v2 rescaled {fmt(spearman([rel_rs[nm] for nm in names], iv), 6, 3)}"
         f"   v2 no rescale {fmt(spearman([rel_raw[nm] for nm in names], iv), 6, 3)}")

    c_rs, c_raw = composite(profiles, rel_rs), composite(profiles, rel_raw)
    c_ideal = ctx["c_ideal"]
    emit(f"{'composite weight':<26}{'ideal':>8}{'v1':>8}{'v2_rescaled':>12}{'v2_raw':>8}")
    for nm in names:
        emit(f"{nm:<26}{c_ideal[nm]:>8.4f}{ctx['c_v1'][nm]:>8.4f}{c_rs[nm]:>12.4f}{c_raw[nm]:>8.4f}")
    emit(f"L1 distance of composite weights to ideal:  v1 {ctx['l1_v1']:.4f}   "
         f"v2 rescaled {sum(abs(c_rs[n] - c_ideal[n]) for n in names):.4f}   "
         f"v2 no rescale {sum(abs(c_raw[n] - c_ideal[n]) for n in names):.4f}")
    emit()
    return pur_union


def run_dataset(data_path: Path, truth_path: Path, id_col: str) -> None:
    df = pd.read_csv(data_path, dtype=str, keep_default_na=False).replace("", None)
    tdf = pd.read_csv(truth_path, dtype=str, keep_default_na=False)
    truth = dict(zip(tdf[id_col], tdf[TRUTH_COLUMN]))
    records = df.to_dict(orient="records")

    # Same setup as learn_rules.
    cdf = pd.DataFrame(records)
    cdf.columns = [str(c).strip() for c in cdf.columns]
    cdf = cdf.where(pd.notna(cdf), None)
    cls = classify_columns(cdf)
    rows = cdf.to_dict(orient="records")
    content = [h for h in cdf.columns if h in set(cls["content_columns"])]
    profiles = build_profiles(content, rows, set(cls["metadata_columns"]))
    normalize_discriminative_weights(profiles)
    lo_hi = {p.name: numeric_series_stats(p.values) for p in profiles}
    names = [p.name for p in profiles]
    ids = [str(r[id_col]).strip() for r in rows]

    pairs = sample_pairs(len(rows), PAIR_SAMPLE, SEED)
    cand_map = _build_candidates(rows, profiles)
    cand_pairs = sorted(cand_map)
    capped = len(cand_pairs) > CANDIDATE_CAP

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, rid in enumerate(ids):
        if rid in truth:
            groups[truth[rid]].append(idx)
    dups = [pr for g in groups.values() for pr in combinations(g, 2)]

    # Ideal reliability and ideal composite weights.
    ideal: dict[str, float] = {}
    for p in profiles:
        sims = [field_similarity(rows[i][p.name], rows[j][p.name], p, lo_hi[p.name])
                for i, j in dups
                if not is_null(rows[i][p.name]) and not is_null(rows[j][p.name])]
        ideal[p.name] = float(np.mean(sims)) if sims else float("nan")

    # v1 and the public-API cross-check.
    rel_v1 = learn_reliability(pairs, rows, profiles, lo_hi)
    c_v1 = composite(profiles, rel_v1)
    api = {v: learn_rules(records, reliability=v)["columns"] for v in ("v1", "v2")}
    ok_v1 = all(abs(api["v1"][n]["reliability_score"] - rel_v1[n]) < 1e-6 for n in names)
    rel_v2_rs, _ = learn_reliability_v2(cand_pairs, rows, profiles, lo_hi)
    ok_v2 = all(abs(api["v2"][n]["reliability_score"] - rel_v2_rs[n]) < 1e-6 for n in names)
    ok_c = all(abs(api["v1"][n]["composite_weight"] - c_v1[n]) < 1e-6 for n in names)

    c_ideal = composite(profiles, {n: (ideal[n] if ideal[n] == ideal[n] else 1.0) for n in names})

    emit("=" * 100)
    emit(f"Dataset: {data_path}  records={len(rows)}  id column={id_col}  seed={SEED}")
    emit(f"Content columns: {names}")
    emit(f"Sampled pairs: {len(pairs)}  candidate pairs: {len(cand_pairs)}"
         f"{' (CAPPED: tool and learn_rules sample differently)' if capped else ''}  "
         f"true duplicate pairs: {len(dups)}")
    emit(f"Cross-check against learn_rules(): v1 reliability {'OK' if ok_v1 else 'MISMATCH'}, "
         f"v1 composite {'OK' if ok_c else 'MISMATCH'}, v2 rescaled reliability "
         f"{'OK' if ok_v2 else 'MISMATCH'}")
    emit("Ideal = mean field_similarity over true duplicate pairs, both values non-null.")
    emit()

    # v1 section.
    v1_anchors = [(i, j) for i, j in pairs
                  if any(exact_match(rows[i].get(p.name), rows[j].get(p.name), p, lo_hi[p.name])
                         for p in profiles)]
    n1, pur1 = purity(v1_anchors, ids, truth)
    emit("--- v1 (sampled pairs with at least one exact-matching column; 0.1-1.0 rescale) ---")
    emit(f"Anchor pairs: {n1}  purity vs truth: " + (f"{pur1:.1%}" if n1 else "n/a"))
    emit(f"{'column':<26}{'anchors':>8}{'ideal':>8}{'v1_rel':>8}")
    for p in profiles:
        n_col = sum(1 for i, j in v1_anchors
                    if not is_null(rows[i][p.name]) and not is_null(rows[j][p.name]))
        emit(f"{p.name:<26}{n_col:>8}{fmt(ideal[p.name])}{fmt(rel_v1[p.name])}")
    iv = [ideal[n] for n in names]
    emit(f"Spearman vs ideal:  v1 {fmt(spearman([rel_v1[n] for n in names], iv), 6, 3)}")
    emit()

    ctx = dict(profiles=profiles, rows=rows, lo_hi=lo_hi, ids=ids, truth=truth,
               ideal=ideal, cand_pairs=cand_pairs, c_ideal=c_ideal, c_v1=c_v1,
               l1_v1=sum(abs(c_v1[n] - c_ideal[n]) for n in names))
    pur2 = v2_section("v2, agreement on at least 1 other column", 1, ctx)
    if pur2 != pur2 or pur2 < PURITY_FLOOR:
        emit(f"v2 anchor purity is {'undefined' if pur2 != pur2 else f'{pur2:.1%}'}, "
             f"under {PURITY_FLOOR:.0%}. Requiring agreement on at least 2 other columns.")
        emit()
        v2_section("v2, agreement on at least 2 other columns", 2, ctx)

    emit(f"--- v1 anchor pairs ({min(len(v1_anchors), MAX_V1_ANCHORS_SHOWN)} of {len(v1_anchors)}) ---")
    for i, j in v1_anchors[:MAX_V1_ANCHORS_SHOWN]:
        agree = [p.name for p in profiles
                 if exact_match(rows[i].get(p.name), rows[j].get(p.name), p, lo_hi[p.name])]
        same = ids[i] in truth and ids[j] in truth and truth[ids[i]] == truth[ids[j]]
        emit(f"{ids[i]} / {ids[j]}  true duplicates: {same}  exact-matching columns: {agree}")
        for tag, k in (("A", i), ("B", j)):
            emit(f"    {tag}: " + " | ".join(f"{p.name}={rows[k].get(p.name)!r}" for p in profiles))
    emit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs=3, action="append", required=True,
                    metavar=("DATA_CSV", "TRUTH_CSV", "ID_COLUMN"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    for data, truth, idc in a.dataset:
        run_dataset(Path(data), Path(truth), idc)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
