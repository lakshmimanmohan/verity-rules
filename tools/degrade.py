"""
Make a degraded copy of a CSV: blank out some values and add typos to others.
Row order, record ids and the truth file are untouched, so the same truth applies.

  python tools/degrade.py data/customers.csv out.csv --seed 42 \
      --blank email,phone 0.5 --typo first_name,last_name,address 0.3

--blank COLS FRACTION: blank that share of ALL rows in each listed column (rows chosen at random).
--typo  COLS FRACTION: change that share of the NON-EMPTY values left in each listed column
                       (after blanking) by one random edit: swap, drop, insert or replace a letter.
COLS is a comma-separated list, or ALL for every column except those in --except.
Both flags may be repeated. Blanking is applied before typos.
"""
from __future__ import annotations

import argparse
import random
import string
import sys
from pathlib import Path

import pandas as pd


def typo(value: str, rng: random.Random) -> str:
    """One random single-character edit. Always returns a different string when possible."""
    if len(value) < 2:
        return value + rng.choice(string.ascii_lowercase)
    for _ in range(10):
        pos = rng.randrange(len(value))
        op = rng.choice(["swap", "drop", "insert", "replace"])
        if op == "swap":
            pos = min(pos, len(value) - 2)
            out = value[:pos] + value[pos + 1] + value[pos] + value[pos + 2:]
        elif op == "drop":
            out = value[:pos] + value[pos + 1:]
        elif op == "insert":
            out = value[:pos] + rng.choice(string.ascii_lowercase) + value[pos:]
        else:
            out = value[:pos] + rng.choice(string.ascii_lowercase) + value[pos + 1:]
        if out != value:
            return out
    return value + "x"


def resolve_cols(spec: str, columns: list[str], skip: set[str]) -> list[str]:
    cols = [c for c in columns if c not in skip] if spec == "ALL" else [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in cols if c not in columns]
    if unknown:
        raise SystemExit("no such column: " + ", ".join(unknown))
    return cols


def degrade(df: pd.DataFrame, seed: int, blank: list[tuple[str, float]], typos: list[tuple[str, float]],
            skip: set[str]) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.copy()
    n = len(out)
    cols = list(out.columns)
    for spec, frac in blank:
        for c in resolve_cols(spec, cols, skip):
            for i in sorted(rng.sample(range(n), round(frac * n))):
                out.at[i, c] = ""
    for spec, frac in typos:
        for c in resolve_cols(spec, cols, skip):
            filled = [i for i in range(n) if out.at[i, c] != ""]
            for i in sorted(rng.sample(filled, round(frac * len(filled)))):
                out.at[i, c] = typo(out.at[i, c], rng)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blank", nargs=2, action="append", default=[], metavar=("COLS", "FRACTION"))
    ap.add_argument("--typo", nargs=2, action="append", default=[], metavar=("COLS", "FRACTION"))
    ap.add_argument("--except", dest="skip", default="", help="columns ALL leaves alone (comma-separated)")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    skip = {c.strip() for c in args.skip.split(",") if c.strip()}
    blank = [(c, float(f)) for c, f in args.blank]
    typos = [(c, float(f)) for c, f in args.typo]
    for _, f in blank + typos:
        if not 0 <= f <= 1:
            raise SystemExit("fractions must be between 0 and 1")
    out = degrade(df, args.seed, blank, typos, skip)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    changed = int((out != df).sum().sum())
    print(f"Wrote {args.out} ({len(out)} rows, {changed} cells changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
