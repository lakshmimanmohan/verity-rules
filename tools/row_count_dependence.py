"""
How much do the learned weights depend on the number of rows? Not shipped to users. Report only.

  .venv/Scripts/python.exe tools/row_count_dependence.py   # writes validation/row_count_dependence.md

1. Random row subsets of customers (10, 25, 50, 75, 100 percent), 3 draws each, seed 42.
2. Group-aware subsets: rows that share a value in a candidate source column (a column whose uniqueness
   ratio is at least CANDIDATE_MIN_UNIQUENESS; values shared by 2 to 10 rows, as in candidate building)
   are kept together, and whole groups are sampled until the row count matches.
   For both: per column composite_weight, anchor count, candidate pairs, largest absolute weight difference
   from the full-data weights and the Spearman rank correlation of the weights.
3. A 20,000-row customers file (10,000 entities) through the CLI: run time and peak memory, with and
   without --max-rows 5000.
No truth file is read here.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import table  # noqa: E402
from verity_rules.cli import load_records  # noqa: E402
from verity_rules.core import (  # noqa: E402
    CANDIDATE_GROUP_MAX, CANDIDATE_GROUP_MIN, CANDIDATE_MIN_UNIQUENESS, learn_rules)
from verity_rules.rule_learner import is_null  # noqa: E402

SEED = 42
FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
DRAWS = 3
CSV = ROOT / "data" / "customers.csv"
BIG_CSV = ROOT / "data" / "customers_20k.csv"
OUT = ROOT / "validation" / "row_count_dependence.md"

# Runs the CLI in a fresh process and prints peak working set (Windows) as the last line.
RUNNER = r"""
import ctypes, ctypes.wintypes as w, sys
from verity_rules.cli import main
rc = main(sys.argv[1:])
class PMC(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]
pmc = PMC(); pmc.cb = ctypes.sizeof(pmc)
k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(k.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb)
print("PEAK_WS_BYTES", pmc.PeakWorkingSetSize)
sys.exit(rc)
"""


def run(records, **kw):
    return learn_rules(records, seed=SEED, debug=True, **kw)


def summary(res: dict) -> dict:
    return {"weights": {c: v["composite_weight"] for c, v in res["columns"].items()},
            "anchors": res["reliability_debug"]["anchor_count"],
            "n_cand": res["candidate_pairs"], "n_rows": res["n_records"]}


def compare(s: dict, full: dict) -> tuple[float, float]:
    common = [c for c in full["weights"] if c in s["weights"]]
    diff = max(abs(s["weights"][c] - full["weights"][c]) for c in common)
    a, b = [s["weights"][c] for c in common], [full["weights"][c] for c in common]
    rho = float(spearmanr(a, b).statistic) if len(set(a)) > 1 else float("nan")
    return diff, rho


def build_groups(records: list[dict], source_cols: list[str]) -> list[list[int]]:
    """Connected components of rows linked by a shared value in a candidate source column."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in source_cols:
        by_value: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            v = r.get(c)
            if not is_null(v):
                by_value.setdefault(str(v).strip().lower(), []).append(i)
        for members in by_value.values():
            if CANDIDATE_GROUP_MIN <= len(members) <= CANDIDATE_GROUP_MAX:
                for m in members[1:]:
                    parent[find(m)] = find(members[0])
    comps: dict[int, list[int]] = {}
    for i in range(len(records)):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=lambda g: g[0])


def group_sample(groups: list[list[int]], target: int, rng: random.Random) -> list[int]:
    order = list(range(len(groups)))
    rng.shuffle(order)
    keep: list[int] = []
    for k in order:
        if len(keep) + len(groups[k]) <= target:
            keep += groups[k]
        if len(keep) == target:
            break
    return sorted(keep)


def sample_tables(label: str, samples: list[tuple[str, dict]], full: dict, cols: list[str]) -> list[str]:
    rows, arows = [], []
    for name, s in samples:
        diff, rho = compare(s, full)
        rows.append([name, s["n_rows"], s["n_cand"], f"{diff:.6f}", f"{rho:.3f}"]
                    + [f"{s['weights'][c]:.4f}" if c in s["weights"] else "excluded" for c in cols])
        arows.append([name] + [s["anchors"].get(c, 0) for c in cols])
    L = [f"### {label}: composite_weight", ""]
    L += table(["Sample", "Rows", "Candidate pairs", "Max abs diff vs full", "Spearman vs full"] + cols, rows)
    L += [f"### {label}: anchor count per column", ""]
    L += table(["Sample"] + cols, arows)
    return L


def cli_run(csv: Path, out_dir: Path, extra: list[str]) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", RUNNER, "learn", str(csv), "--out", str(out_dir), *extra],
                       capture_output=True, text=True, cwd=ROOT)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    peak = int([ln for ln in p.stdout.splitlines() if ln.startswith("PEAK_WS_BYTES")][-1].split()[1])
    rules = json.loads((out_dir / "rules.json").read_text(encoding="utf-8"))
    return {"seconds": wall, "peak_mb": peak / 2**20, "rows": rules["n_records"],
            "cands": rules["run"]["n_candidate_pairs"], "cols": len(rules["columns"])}


def main() -> int:
    records, n = load_records(CSV, 20_000, SEED)
    full_res = run(records)
    full = summary(full_res)
    cols = sorted(full["weights"], key=lambda c: (-full["weights"][c], c))
    L = ["# Row-count dependence of the learned weights", "",
         f"Dataset: `{CSV.name}`, {n:,} rows. Seed {SEED} for the draws and for learn_rules. Report only; nothing tuned.",
         "Columns are ordered by full-data weight. Max abs diff and Spearman compare each sample's weights "
         "with the full-data weights, over the columns kept in both.", "",
         "## Full data", ""]
    L += table(["Rows", "Candidate pairs"] + cols,
               [[n, full["n_cand"]] + [f"{full['weights'][c]:.4f}" for c in cols]])
    L += table(["Anchors"] + cols, [[""] + [full["anchors"].get(c, 0) for c in cols]])

    rng = random.Random(SEED)
    rand_samples = []
    for f in FRACTIONS:
        k = round(n * f)
        for d in range(1, DRAWS + 1):
            keep = sorted(rng.sample(range(n), k))
            rand_samples.append((f"{f:.0%} draw {d}", summary(run([records[i] for i in keep]))))
            print("random", f, d, flush=True)
    L += ["## Random row subsets", "",
          f"{DRAWS} draws per fraction, rows drawn uniformly without replacement. At 100% every draw is the full file.", ""]
    L += sample_tables("Random", rand_samples, full, cols)

    source_cols = [c for c, v in full_res["columns"].items() if v["uniqueness_ratio"] >= CANDIDATE_MIN_UNIQUENESS]
    groups = build_groups(records, source_cols)
    sizes = sorted((len(g) for g in groups), reverse=True)
    multi = [s for s in sizes if s > 1]
    rng = random.Random(SEED)
    grp_samples = []
    for f in FRACTIONS:
        k = round(n * f)
        for d in range(1, DRAWS + 1):
            keep = group_sample(groups, k, rng)
            grp_samples.append((f"{f:.0%} draw {d}", summary(run([records[i] for i in keep]))))
            print("group", f, d, flush=True)
    L += ["## Group-aware row subsets", "",
          f"Candidate source columns (uniqueness ratio >= {CANDIDATE_MIN_UNIQUENESS}): {', '.join(source_cols)}. "
          f"Rows sharing a value in any of them (a value shared by {CANDIDATE_GROUP_MIN} to {CANDIDATE_GROUP_MAX} rows) are "
          f"linked, and linked rows form a group that is kept or dropped whole. {len(groups):,} groups: "
          f"{len(groups) - len(multi):,} single rows and {len(multi):,} groups of 2 or more rows "
          f"(largest {sizes[0]}, median among multi-row groups {multi[len(multi) // 2] if multi else 0}). "
          "Groups are shuffled with the seed and added whole while they fit the target row count; groups that "
          "do not fit are skipped, so the row count matches exactly.", ""]
    L += sample_tables("Group-aware", grp_samples, full, cols)

    L += ["## Run time and peak memory, 20,000-row customers file", ""]
    if not BIG_CSV.is_file():
        L += [f"`{BIG_CSV.name}` not found. Generate it with tools/generate_data.py (NUM_ENTITIES 10000, TOTAL_ROWS 20000) "
              "and tools/split_truth.py first.", ""]
    else:
        runs = []
        for label, extra in (("default (--max-rows 20000)", []), ("--max-rows 5000", ["--max-rows", "5000"])):
            r = cli_run(BIG_CSV, ROOT / "out" / "rowcount_20k" / label.split()[-1].strip("()"), extra)
            runs.append([label, r["rows"], r["cols"], r["cands"], f"{r['seconds']:.1f}", f"{r['peak_mb']:.0f}"])
            print("cli", label, r, flush=True)
        L += ["`verity learn data/customers_20k.csv` in a fresh process (so the time includes Python start and "
              "imports). Peak memory is the process peak working set on Windows.", ""]
        L += table(["Run", "Rows analysed", "Columns", "Candidate pairs", "Seconds", "Peak memory (MB)"], runs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
