"""Command-line interface: `verity learn data.csv --out rules/`.

This is the only module in the package allowed to read or write files.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .core import learn_rules

DEFAULT_SEED = 42
DEFAULT_MAX_ROWS = 20_000
DEBUG_KEYS = ("reliability_debug", "threshold", "similarity_stats")
CATEGORY_ORDER = ("likely_match", "unclear", "likely_non_match")
MASK = "***"
PAIRS_NOTICE = ("Note: calibration_pairs.csv and rules.json contain values from the source data. "
                "Handle them like the source data.")


class CliError(Exception):
    """A user-facing problem; printed as one line, no traceback."""


def load_records(path: Path, max_rows: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    """Read a CSV as strings (empty cells become None). Returns (records, original row count).
    Above max_rows, a fixed-seed sample is taken and original row order is kept."""
    if not path.is_file():
        raise CliError(f"file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", None)
    except pd.errors.EmptyDataError:
        raise CliError(f"{path} is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise CliError(f"could not read {path} as CSV: {exc}") from None
    n_original = len(df)
    if n_original < 2:
        raise CliError(f"{path} has {n_original} data rows; Verity needs at least 2.")
    if n_original > max_rows:
        df = df.sample(n=max_rows, random_state=seed).sort_index()
    return df.to_dict(orient="records"), n_original


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


SEPARABILITY_TEXT = {
    "clear": ("Likely duplicates score far above random pairs. Few pairs should land in "
              "the grey zone, so manual review effort is low."),
    "moderate": ("Likely duplicates score above random pairs, but the two groups overlap "
                 "somewhat. Expect to review a real band of in-between pairs when you "
                 "choose a cutoff."),
    "weak": ("Likely duplicates score only slightly above random pairs. Many pairs will "
             "sit near any cutoff, so expect heavy review effort and treat the weights as "
             "a rough guide."),
}


def build_report(result: dict[str, Any], source: Path, n_original: int,
                 anchor_counts: dict[str, int] | None) -> str:
    cols = result["columns"]
    ordered = sorted(cols.items(), key=lambda kv: (-kv[1]["composite_weight"], kv[0]))
    sep = result["separability"]
    n = result["n_records"]
    L: list[str] = []
    L += [f"# Verity rules report: {source.name}", "",
          "## What is Verity?", "",
          "Verity reads a dataset and learns which columns help identify the same real-world "
          "entity, and how much each column should count. It outputs rules (weights and "
          "comparison methods) and does not match records itself.", "",
          "## Dataset", "",
          f"- Source file: `{source.name}`",
          f"- Rows in file: {n_original:,}",
          f"- Rows analysed: {n:,}" + (" (a random sample; see Notes)" if n < n_original else ""),
          f"- Record id column: `{result['record_id_column']}`",
          f"- Record pairs sampled: {result['n_pairs_sampled']:,}", "",
          "## Columns, by weight", "",
          "| Column | Type | Comparison method | Weight | Reliability | Anchors | Reason |",
          "|---|---|---|---|---|---|---|"]
    for name, c in ordered:
        anchors = "n/a" if anchor_counts is None else anchor_counts.get(name, 0)
        L.append(f"| {_md_cell(name)} | {c['semantic_type']} | {c['comparison_method']} | "
                 f"{c['composite_weight']:.3f} | {c['reliability_score']:.3f} | {anchors} | "
                 f"{_md_cell(c['reason'])} |")
    L += ["", "Anchors are the likely-duplicate pairs used to measure that column's reliability.", "",
          "## Excluded columns", ""]
    if result["columns_excluded"]:
        L += ["| Column | Reason |", "|---|---|"]
        L += [f"| {_md_cell(e['name'])} | {_md_cell(e['reason'])} |"
              for e in result["columns_excluded"]]
    else:
        L.append("No columns were excluded.")
    L += ["", "## Separability", ""]
    if sep["label"] is None:
        L.append("Separability could not be computed because no likely-duplicate candidate "
                 "pairs were found.")
    else:
        L += [f"**{sep['label'].capitalize()}** (gap {sep['gap']:.3f}).", "",
              "Separability compares how alike likely duplicates look with how alike random "
              "pairs look, using the learned weights. The gap is the difference between those "
              "two averages.", "",
              f"- Clear (gap 0.30 or more): groups are well apart, little review effort.",
              f"- Moderate (0.15 to 0.30): some overlap, moderate review effort.",
              f"- Weak (under 0.15): little separation, heavy review effort.", "",
              SEPARABILITY_TEXT[sep["label"]]]
    L += ["", "## Notes", ""]
    L += [f"- {_md_cell(t)}" for t in result["notes"]] or ["None."]
    L += ["", "## How to use these rules", "",
          "- **Weights** show how much each column should count. They sum to 1. They "
          "describe the data, so you can use them in any matching tool.",
          "- **Reliability** is how similar a column's values stay across records that likely "
          "refer to the same thing.",
          "- **Comparison method** says how to compare two values of that column.",
          "- **Match cutoff:** set your own in your own tool. Score the pairs in "
          "`calibration_pairs.csv` there, fill in the `your_label` column, and pick the "
          "cutoff that best separates your labels. Verity does not provide one.", "",
          "## Limits", "",
          "- Verity does not match or clean records.",
          "- The weights come from one run on a sample of the data. Rerun if the data changes.",
          ""]
    return "\n".join(L)


def mask_pairs(result: dict[str, Any], mask: set[str]) -> None:
    """Replace non-empty values of the masked columns with *** in the pairs, in place.
    Empty values stay empty. Masking the record id column masks the pair record ids."""
    if not mask:
        return
    for pairs in result["calibration_pairs"].values():
        for pair in pairs:
            for side in ("a", "b"):
                vals = pair[f"values_{side}"]
                for c in vals:
                    if c in mask and vals[c] is not None:
                        vals[c] = MASK
                if result["record_id_column"] in mask and pair[f"record_id_{side}"]:
                    pair[f"record_id_{side}"] = MASK


def shuffled_pairs(result: dict[str, Any], seed: int) -> list[tuple[str, dict[str, Any]]]:
    """All calibration pairs as (suggested_label, pair), in a seeded random order so the
    row order of the pairs file does not reveal the label."""
    flat = [(cat, pair) for cat in CATEGORY_ORDER
            for pair in result["calibration_pairs"].get(cat, [])]
    random.Random(seed).shuffle(flat)
    return flat


def write_calibration_csv(path: Path, key_path: Path, result: dict[str, Any], seed: int) -> int:
    """Write the shuffled pairs file (no labels) and the key file (pair_id, suggested_label)."""
    content = list(result["columns"])
    header = ["pair_id", "record_id_a", "record_id_b"]
    for c in content:
        header += [f"{c}_a", f"{c}_b"]
    header.append("your_label")
    flat = shuffled_pairs(result, seed)
    with path.open("w", newline="", encoding="utf-8") as fh, \
            key_path.open("w", newline="", encoding="utf-8") as kh:
        w, kw = csv.writer(fh), csv.writer(kh)
        w.writerow(header)
        kw.writerow(["pair_id", "suggested_label"])
        for k, (cat, pair) in enumerate(flat, start=1):
            row = [k, pair["record_id_a"], pair["record_id_b"]]
            for c in content:
                row += [pair["values_a"].get(c), pair["values_b"].get(c)]
            row.append("")
            w.writerow(["" if v is None or v != v else v for v in row])  # v != v: NaN
            kw.writerow([k, cat])
    return len(flat)


def cmd_learn(args: argparse.Namespace) -> int:
    src, out = Path(args.csv), Path(args.out)
    if args.max_rows < 2:
        raise CliError("--max-rows must be at least 2.")
    records, n_original = load_records(src, args.max_rows, args.seed)
    try:
        result = learn_rules(records, seed=args.seed, debug=True)
    except ValueError as exc:
        if "record id" in str(exc).lower():
            raise CliError(f"no usable record id column found in {src}.") from None
        raise CliError(str(exc)) from None

    debug = {k: result.pop(k) for k in DEBUG_KEYS if k in result}
    if n_original > len(records):
        result["notes"].append(
            f"Dataset has {n_original:,} rows; a random sample of {len(records):,} "
            f"(seed {args.seed}) was analysed.")
    anchors = debug.get("reliability_debug", {}).get("anchor_count")

    mask = {c.strip() for c in args.mask_columns.split(",") if c.strip()} if args.mask_columns else set()
    known = (set(result["columns"]) | {e["name"] for e in result["columns_excluded"]}
             | {result["record_id_column"]})
    unknown = sorted(mask - known)
    if unknown:
        raise CliError("--mask-columns: no such column: " + ", ".join(unknown))
    if args.no_pairs:
        result["calibration_pairs"] = {cat: [] for cat in CATEGORY_ORDER}
    mask_pairs(result, mask)

    out.mkdir(parents=True, exist_ok=True)
    (out / "rules.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (out / "rules_report.md").write_text(
        build_report(result, src, n_original, anchors), encoding="utf-8")
    written = ["rules.json", "rules_report.md"]
    n_pairs = 0
    if not args.no_pairs:
        n_pairs = write_calibration_csv(out / "calibration_pairs.csv",
                                        out / "calibration_pairs_key.csv", result, args.seed)
        written += ["calibration_pairs.csv", "calibration_pairs_key.csv"]
    if args.debug:
        (out / "debug.json").write_text(json.dumps(debug, indent=2, allow_nan=False), encoding="utf-8")
        written.append("debug.json")
    print(f"Learned rules for {len(records):,} rows, {len(result['columns'])} columns, "
          f"{n_pairs} calibration pairs.")
    print(f"Wrote {', '.join(written)} to {out}")
    if not args.no_pairs:
        print(PAIRS_NOTICE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verity", description="Learn entity-matching rules from a CSV.")
    sub = p.add_subparsers(dest="command", required=True)
    learn = sub.add_parser("learn", help="learn column weights and rules from a CSV")
    learn.add_argument("csv", help="path to the input CSV")
    learn.add_argument("--out", required=True, help="output folder (created if missing)")
    learn.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default 42)")
    learn.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                       help="sample down to this many rows if the file is larger (default 20000)")
    learn.add_argument("--mask-columns", metavar="COLS",
                       help="comma-separated column names; their values become *** in the pairs "
                            "(calibration_pairs.csv and rules.json)")
    learn.add_argument("--no-pairs", action="store_true",
                       help="write no calibration pairs files and leave calibration_pairs empty")
    learn.add_argument("--debug", action="store_true", help="also write debug.json")
    learn.set_defaults(func=cmd_learn)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
