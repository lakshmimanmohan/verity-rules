"""
Compare expected_pair_coverage with the measured coverage of true duplicate pairs. Not shipped to users.

  .venv/Scripts/python.exe tools/check_coverage.py

Coverage for a pair = sum of weights of columns present in both records / sum of all weights.
Datasets: customers and the degraded copies C, C25 and H. Truth files are read here only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import Dataset, table  # noqa: E402
from verity_rules.rule_learner import is_null  # noqa: E402

DATA = ROOT / "data"
COPIES = [("customers", "customers.csv"), ("Copy C", "customers_degraded_C.csv"),
          ("Copy C25", "customers_degraded_C25.csv"), ("Copy H", "customers_degraded_H.csv")]


def measured_coverage(ds: Dataset) -> tuple[float, float, int]:
    w = {c: ds.result["columns"][c]["composite_weight"] for c in ds.columns}
    total = sum(w.values())
    covs = []
    for i, j in ds.dup_pairs:
        present = sum(w[c] for c in ds.columns
                      if not is_null(ds.rows[i].get(c)) and not is_null(ds.rows[j].get(c)))
        covs.append(present / total)
    return float(np.mean(covs)), float(np.median(covs)), len(covs)


def main() -> int:
    rows = []
    for name, csv in COPIES:
        ds = Dataset(name, DATA / csv, DATA / "customers_truth.csv")
        mean, med, n = measured_coverage(ds)
        exp = ds.result["expected_pair_coverage"]
        rows.append([name, f"{exp:.6f}", f"{mean:.6f}", f"{med:.6f}", f"{exp - mean:+.6f}", n])
    print("\n".join(table(["Dataset", "expected_pair_coverage", "Measured mean (true duplicate pairs)",
                           "Measured median", "Expected minus measured", "Duplicate pairs"], rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
