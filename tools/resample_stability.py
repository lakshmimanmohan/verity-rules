"""
Evidence for the pair-coverage and calibration changes. Not shipped to users. Report only; nothing is tuned.

  .venv\\Scripts\\python.exe tools/resample_stability.py   # writes validation/stability_and_near_miss.md

1. Resample stability: 20 random 50% row subsets of customers and products (seeded). learn_rules runs on each
   subset with seed 42. Per column: min and max composite_weight, and whether the top three columns by weight
   (as a set, and in order) are the same in every subset.
2. Near-miss truth share: of all pairs that are not candidate pairs, the share that are true duplicates,
   and how many of the calibration near-miss pairs (source non_candidate_top) are true duplicates.
3. Bigram check: five true duplicate pairs and five random pairs of one column (default member_part_number)
   with their bigram similarity, using the column's own comparison method.
Truth files are read here only; verity_rules/ never reads them.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import Dataset, SEED, pct, table  # noqa: E402
from verity_rules.core import _build_candidates, learn_rules  # noqa: E402
from verity_rules.rule_learner import field_similarity, is_null  # noqa: E402

DATASETS = ("customers", "products")
N_SUBSETS = 20
TOP_K = 3
N_EXAMPLES = 5


def top_k(weights: dict[str, float]) -> list[str]:
    return sorted(weights, key=lambda c: (-weights[c], c))[:TOP_K]


def resample_section(ds: Dataset) -> list[str]:
    n = len(ds.records)
    rng = random.Random(SEED)
    runs = []
    for _ in range(N_SUBSETS):
        keep = sorted(rng.sample(range(n), n // 2))
        res = learn_rules([ds.records[i] for i in keep], seed=SEED)
        runs.append({c: v["composite_weight"] for c, v in res["columns"].items()})
    rows = []
    for c in ds.columns:
        ws = [r[c] for r in runs if c in r]
        rows.append([c, f"{min(ws):.6f}", f"{max(ws):.6f}", f"{max(ws) - min(ws):.6f}",
                     f"{len(ws)} of {N_SUBSETS}"])
    tops = [top_k(r) for r in runs]
    same_set = len({frozenset(t) for t in tops}) == 1
    same_order = len({tuple(t) for t in tops}) == 1
    full = top_k({c: ds.result["columns"][c]["composite_weight"] for c in ds.columns})
    L = [f"### Resample stability, {ds.name}", "",
         f"{N_SUBSETS} random 50% row subsets ({n // 2} of {n} rows), seed {SEED} for the draw and for learn_rules. "
         "Columns present in a subset run are counted in the last column.", ""]
    L += table(["Column", "Min weight", "Max weight", "Range", "Runs with column"], rows)
    L.append(f"Top {TOP_K} columns on the full data: {', '.join(full)}.")
    L.append(f"Top {TOP_K} the same set in every subset: **{'yes' if same_set else 'no'}**. "
             f"Same order in every subset: **{'yes' if same_order else 'no'}**.")
    if not same_order:
        counts: dict[tuple[str, ...], int] = {}
        for t in tops:
            counts[tuple(t)] = counts.get(tuple(t), 0) + 1
        L += [f"- {v} runs: {' > '.join(k)}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return L + [""]


def near_miss_section(ds: Dataset) -> list[str]:
    n = len(ds.rows)
    cands = set(_build_candidates(ds.rows, ds.profiles))
    total = n * (n - 1) // 2
    dups = set(ds.dup_pairs)
    dups_non_cand = len(dups - cands)
    non_cand = total - len(cands)
    picked = [p for p in ds.result["calibration_pairs"]["unclear"] if p["source"] == "non_candidate_top"]
    hits = sum(ds.is_dup(ds.index[p["record_id_a"]], ds.index[p["record_id_b"]]) for p in picked)
    mid = [p for p in ds.result["calibration_pairs"]["unclear"] if p["source"] == "candidate_mid"]
    mid_hits = sum(ds.is_dup(ds.index[p["record_id_a"]], ds.index[p["record_id_b"]]) for p in mid)
    L = [f"### Non-candidate pairs, {ds.name}", ""]
    L += table(["Measure", "Value"], [
        ["Records", n],
        ["All pairs", f"{total:,}"],
        ["Candidate pairs", f"{len(cands):,}"],
        ["Non-candidate pairs", f"{non_cand:,}"],
        ["True duplicate pairs (all)", f"{len(dups):,}"],
        ["True duplicate pairs that are candidates", f"{len(dups & cands):,} ({pct(len(dups & cands) / len(dups))})"],
        ["True duplicate pairs among non-candidates", f"{dups_non_cand:,}"],
        ["Share of non-candidate pairs that are true duplicates", f"{100 * dups_non_cand / non_cand:.4f}%"],
        ["Near-miss pairs in unclear (non_candidate_top)", len(picked)],
        ["Near-miss pairs that are true duplicates", f"{hits} ({pct(hits / len(picked) if picked else None)})"],
        ["Mid-range candidate pairs in unclear (candidate_mid)", len(mid)],
        ["Mid-range candidate pairs that are true duplicates",
         f"{mid_hits} ({pct(mid_hits / len(mid) if mid else None)})"],
    ])
    return L


def bigram_section(ds: Dataset, column: str) -> list[str]:
    p = next((q for q in ds.profiles if q.name == column), None)
    if p is None:
        return [f"### Bigram examples, {ds.name}", "", f"Column `{column}` is not a kept column here.", ""]
    method = ds.result["columns"][column]["comparison_method"]
    ok = lambda i, j: not is_null(ds.rows[i].get(column)) and not is_null(ds.rows[j].get(column))  # noqa: E731

    def row(kind: str, i: int, j: int) -> list:
        a, b = ds.rows[i][column], ds.rows[j][column]
        s = field_similarity(a, b, p, ds.lo_hi[column], comparator_variant=ds.variant)
        return [kind, f"`{a}`", f"`{b}`", f"{s:.6f}"]

    rng = random.Random(SEED)
    dups = [d for d in ds.dup_pairs if ok(*d)]
    dup_pick = rng.sample(dups, min(N_EXAMPLES, len(dups)))
    rand_pick, n = [], len(ds.rows)
    while len(rand_pick) < N_EXAMPLES:
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j and not ds.is_dup(i, j) and ok(i, j):
            rand_pick.append((i, j))
    L = [f"### Bigram examples, {ds.name}.{column}", "",
         f"Comparison method: {method}. Similarity is the share of adjacent character pairs both values share "
         f"(after lower-casing and dropping everything except letters and digits). Pairs drawn with seed {SEED}.", ""]
    L += table(["Pair", "Value A", "Value B", "Similarity"],
               [row("true duplicate", i, j) for i, j in dup_pick] + [row("random", i, j) for i, j in rand_pick])
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "validation" / "stability_and_near_miss.md")
    ap.add_argument("--bigram-column", default="member_part_number")
    ap.add_argument("--bigram-dataset", default="products")
    args = ap.parse_args(argv)
    data = ROOT / "data"
    L = ["# Stability and near-miss evidence", "",
         "Report only. No weights, cutoffs or rules were changed. Seed 42. Generated by `tools/resample_stability.py`.", ""]
    sets = {}
    for name in DATASETS:
        sets[name] = Dataset(name, data / f"{name}.csv", data / f"{name}_truth.csv")
        print(f"loaded {name}", flush=True)
    L += ["## Resample stability", ""]
    for ds in sets.values():
        L += resample_section(ds)
        print(f"resampled {ds.name}", flush=True)
    L += ["## Share of true duplicates among non-candidate pairs", ""]
    for ds in sets.values():
        L += near_miss_section(ds)
    L += ["## Bigram similarity examples", ""]
    L += bigram_section(sets[args.bigram_dataset], args.bigram_column)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
