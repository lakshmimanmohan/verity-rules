"""
Split a full CSV (with a true_entity_id column) into a stripped CSV and a truth CSV.

Writes <name>.csv (all columns except true_entity_id) and <name>_truth.csv
(record id column + true_entity_id) into the output folder. <name> is the
input file stem with a trailing "_full" removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRUTH_COLUMN = "true_entity_id"


def split_truth(full_path: Path, id_column: str, out_dir: Path) -> tuple[Path, Path]:
    df = pd.read_csv(full_path, dtype=str, keep_default_na=False)
    for col in (id_column, TRUTH_COLUMN):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in {full_path}")

    name = full_path.stem.removesuffix("_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{name}.csv"
    truth_path = out_dir / f"{name}_truth.csv"

    df.drop(columns=[TRUTH_COLUMN]).to_csv(data_path, index=False)
    df[[id_column, TRUTH_COLUMN]].to_csv(truth_path, index=False)
    return data_path, truth_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a full CSV into a stripped CSV and a ground-truth CSV."
    )
    parser.add_argument("full_csv", type=Path, help="Full CSV including true_entity_id")
    parser.add_argument("id_column", help="Record id column name")
    parser.add_argument("out_dir", type=Path, help="Output folder")
    args = parser.parse_args()
    data_path, truth_path = split_truth(args.full_csv, args.id_column, args.out_dir)
    print(f"Wrote {data_path} and {truth_path}")
