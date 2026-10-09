"""Command-line interface: `verity learn data.csv --out rules/`.

This is the only module in the package allowed to read or write files.
"""
from __future__ import annotations

import argparse
import codecs
import csv
import io
import json
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .column_classification import (
    classify_columns, is_idlike_name, pick_record_id_column, record_id_problem,
)
from .core import (
    CANDIDATE_CAP, CANDIDATE_GROUP_MAX, CANDIDATE_GROUP_MIN, CANDIDATE_MIN_UNIQUENESS,
    NO_CANDIDATES_NOTE, USER_EXCLUDED_RULE, _build_candidates, learn_rules,
)
from .rule_learner import (
    ANCHOR_MIN_COUNT, ANCHOR_MIN_UNIQUENESS, build_profiles, column_uniqueness_and_nulls,
    learn_reliability_v2, numeric_series_stats,
)
from .sampling import group_sample_indices

DEFAULT_SEED = 42
DEFAULT_MAX_ROWS = 20_000
SAMPLE_MODES = ("groups", "random")
DEBUG_KEYS = ("reliability_debug", "candidate_pairs", "threshold", "similarity_stats", "separability")
CATEGORY_ORDER = ("likely_match", "unclear", "likely_non_match")
MASK = "***"
ROW_COLUMN = "_row"
OUTPUT_FILES = ("rules.json", "rules_report.md", "columns.csv", "calibration_pairs.csv",
                "calibration_pairs_key.csv")
COLUMN_FIELDS = ("semantic_type", "comparison_method", "uniqueness_ratio", "null_rate",
                 "reliability_score", "composite_weight", "reason")
PAIRS_NOTICE = ("Note: calibration_pairs.csv and rules.json contain values from the source data. "
                "Handle them like the source data.")


class CliError(Exception):
    """A user-facing problem; printed as one line, no traceback."""


DELIMITERS = {",": "comma", ";": "semicolon", "	": "tab", "|": "pipe"}
_DELIMITER_ALIASES = {**{v: k for k, v in DELIMITERS.items()}, "\t": "	"}
SNIFF_LINES = 20


def decode_bytes(raw: bytes, path: Path, encoding: str | None) -> tuple[str, str]:
    """Decode a file. Without an override: utf-8-sig (a BOM is dropped), then cp1252 if the bytes
    are not valid UTF-8. Never ignores or replaces bad bytes. Returns (text, encoding used)."""
    if encoding is not None:
        try:
            codec = codecs.lookup(encoding).name
        except LookupError:
            raise CliError(f"--encoding: unknown encoding: {encoding}") from None
        try:
            return raw.decode("utf-8-sig" if codec == "utf-8" else encoding), encoding
        except UnicodeDecodeError as exc:
            raise CliError(f"{path} is not valid {encoding} (byte {exc.start}); "
                           "check --encoding.") from None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252"), "cp1252"
        except UnicodeDecodeError as exc:
            raise CliError(f"{path} is neither UTF-8 nor cp1252 (byte {exc.start}); "
                           "use --encoding.") from None
    return text, "utf-8-sig" if raw.startswith(codecs.BOM_UTF8) else "utf-8"


def sniff_delimiter(text: str) -> str:
    """Pick comma, semicolon, tab or pipe from the first lines: the one that splits the most
    lines into the same number (at least 2) of fields. Falls back to comma."""
    lines = [ln for ln in text.splitlines()[:SNIFF_LINES] if ln.strip()]
    best, best_key = ",", (0.0, 1)
    for d in DELIMITERS:
        counts = [len(row) for row in csv.reader(lines, delimiter=d)]
        if not counts:
            continue
        modal = max(set(counts), key=lambda c: (counts.count(c), c))
        if modal < 2:
            continue
        key = (counts.count(modal) / len(counts), modal)
        if key > best_key:
            best, best_key = d, key
    return best


def parse_delimiter(value: str) -> str:
    d = _DELIMITER_ALIASES.get(value.lower(), value)
    if len(d) != 1:
        raise CliError(f"--delimiter: use a single character or one of: "
                       f"{', '.join(DELIMITERS.values())}. Got: {value}")
    return d


def delimiter_name(d: str) -> str:
    return DELIMITERS.get(d, repr(d))


def read_table(path: Path, encoding: str | None = None,
               delimiter: str | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Read a CSV as strings (empty cells become None). Returns (records, {"encoding", "delimiter"})
    with what was used, whether given or detected."""
    if not path.is_file():
        raise CliError(f"file not found: {path}")
    delim = parse_delimiter(delimiter) if delimiter is not None else None
    text, used = decode_bytes(path.read_bytes(), path, encoding)
    if delim is None:
        delim = sniff_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delim, dtype=str,
                         keep_default_na=False).replace("", None)
    except pd.errors.EmptyDataError:
        raise CliError(f"{path} is empty.") from None
    except pd.errors.ParserError as exc:
        raise CliError(f"could not read {path} as CSV ({delimiter_name(delim)}-delimited): "
                       f"{exc}") from None
    if df.shape[1] < 2:
        raise CliError(f"only one column was read from {path} with the {delimiter_name(delim)} "
                       "delimiter; check the delimiter (use --delimiter).")
    if len(df) < 2:
        raise CliError(f"{path} has {len(df)} data rows; Verity needs at least 2.")
    return df.to_dict(orient="records"), {"encoding": used, "delimiter": delimiter_name(delim)}


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a CSV as strings (empty cells become None)."""
    return read_table(path)[0]


def sample_records(records: list[dict[str, Any]], max_rows: int, seed: int,
                   mode: str = "groups", exclude: list[str] = (), keep: list[str] = (),
                   record_id_column: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cut records down to max_rows, keeping original row order. mode "groups" keeps rows that
    share a distinctive value together; "random" draws rows uniformly. Returns (records, info)."""
    n = len(records)
    info: dict[str, Any] = {"mode": mode, "rows_in_file": n, "rows_analysed": min(n, max_rows),
                            "groups_kept": 0}
    if n <= max_rows:
        return records, info
    if mode == "random":
        keep = sorted(pd.DataFrame(index=range(n)).sample(n=max_rows, random_state=seed).index)
    else:
        keep, gi = group_sample_indices(records, max_rows, seed, exclude, keep, record_id_column)
        info["groups_kept"] = gi["groups_kept"]
    return [records[i] for i in keep], info


def load_records(path: Path, max_rows: int, seed: int,
                 mode: str = "groups") -> tuple[list[dict[str, Any]], int]:
    """Read and sample a CSV. Returns (records, original row count)."""
    records = read_records(path)
    return sample_records(records, max_rows, seed, mode)[0], len(records)


def choose_record_id(records: list[dict[str, Any]], requested: str | None) -> tuple[list[dict[str, Any]], str, bool]:
    """Decide the record id column for the whole file. Returns (records, column, row_numbers).
    With no usable id column, adds a 1-based row-number column (file order, after the header)
    named _row, or _row_2, _row_3, ... if the file already has that name."""
    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]
    if requested is not None:
        name = requested.strip()
        if name not in df.columns:
            raise CliError(f"--id-column: no such column: {requested}")
        why = record_id_problem(df[name].tolist())
        if why:
            raise CliError(f"--id-column: {name} cannot be the record id because {why}.")
        return records, name, False
    picked = pick_record_id_column(df)
    if picked is not None:
        return records, picked, False
    name, k = ROW_COLUMN, 1
    while name in df.columns:
        k += 1
        name = f"{ROW_COLUMN}_{k}"
    return [{**r, name: str(i)} for i, r in enumerate(records, start=1)], name, True


def _md_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


COVERAGE_LOW = 0.3  # rule of thumb from synthetic tests
ANCHOR_LEVEL = 2  # agreeing columns an anchor needs before the rule is relaxed

_METHOD_GLOSSARY = {
    "email": ("Lower-cases both values, reads `(at)` as `@`, then compares the part before the `@` "
              "and the domain separately, each as the share of distinct characters the two parts "
              "have in common, and averages the two. If either value has no `@`, it compares "
              "the whole values that way."),
    "exact": ("1 if the two values are identical after trimming spaces (capitals and punctuation "
              "matter), otherwise 0."),
    "exact_normalized": ("1 if the two values are identical after lower-casing and dropping "
                         "everything except letters and digits, otherwise 0. `AB-123` and "
                         "`ab 123` match."),
    "bigram_jaccard": ("Lower-cases the values and drops everything except letters and digits, "
                       "then splits each into adjacent character pairs (bigrams). The score is "
                       "the pairs both values share divided by all distinct pairs in either. "
                       "Order matters: `abcd` and `dcba` share no pairs and score 0, and one "
                       "typo costs only the pairs next to it."),
    "text_bigram": ("Lower-cases the values and drops everything except letters and digits, "
                    "then splits each into adjacent character pairs (bigrams). The score is "
                    "the pairs both values share divided by all distinct pairs in either. "
                    "Because pairs are adjacent, order matters: `abcd` and `dcba` share no "
                    "pairs and score 0, and a small typo lowers the score a little instead of failing the match. Used for "
                    "text columns."),
    "text_character": ("The number of distinct characters both values contain, divided by the "
                       "number of distinct characters in either, ignoring capitals and order. "
                       "It is a rough measure: the same letters in a different order score 1. "
                       "Not used by default; it appears only when an older comparator variant "
                       "is selected."),
    "text_hybrid": ("A weighted sum: 0.65 times the cosine similarity of the two records' text "
                    "embeddings, plus 0.35 times the share of distinct characters the two "
                    "values have in common (distinct characters both contain divided by "
                    "distinct characters in either, ignoring capitals and order). It appears "
                    "only when embeddings were used. If a record has no embedding, only the "
                    "character share is used."),
    "date": ("1 minus the number of days apart divided by 3,650 (ten years), never below 0. "
             "A value that cannot be read as a date scores 0."),
    "numeric_range": ("Both numbers are placed on a 0 to 1 scale using the smallest and largest "
                      "value in the column. The score is 1 minus the difference. A value that "
                      "is not a number scores 0."),
}


def _ordered_columns(result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(result["columns"].items(), key=lambda kv: (-kv[1]["composite_weight"], kv[0]))


def _reliability_measured(result: dict[str, Any], anchor_counts: dict[str, int] | None) -> bool:
    """True when some column had enough anchor pairs and reliability is not 1.0 everywhere."""
    if not anchor_counts or not any(n >= ANCHOR_MIN_COUNT for n in anchor_counts.values()):
        return False
    return any(c["reliability_score"] != 1.0 for c in result["columns"].values())


def _read_first_box(result: dict[str, Any]) -> list[str]:
    """The box shown at the top of the report when no candidate pairs were found."""
    if any(c["uniqueness_ratio"] >= ANCHOR_MIN_UNIQUENESS for c in result["columns"].values()):
        why = ("No candidate pairs were found: no value in a column with at least half of its "
               "values distinct is shared by 2 to 10 records.")
    else:
        why = ("No candidate pairs were found: no column has at least half of its values distinct.")
    return ["> **Read this first**", ">",
            f"> {why} Reliability was not measured, so every column's reliability is set to 1.0. "
            "The weights come from distinctiveness only. This usually means each entity appears "
            "in many rows, or duplicates differ in every column. Run verity check on the file. "
            "Verity may not suit it.", ""]


NO_ANCHORS_NOTE = ("No column had enough anchor pairs, so reliability was not measured and is "
                   "set to 1.0 for every column.")
_PER_COLUMN_ANCHOR_NOTE = re.compile(r"only \d+ anchor pairs \(need \d+\)")


def _no_anchors(result: dict[str, Any], anchor_counts: dict[str, int] | None) -> bool:
    """True when anchor counts are known and no kept column reached ANCHOR_MIN_COUNT."""
    return anchor_counts is not None and all(
        anchor_counts.get(name, 0) < ANCHOR_MIN_COUNT for name in result["columns"])


def _report_notes(notes: list[str], no_anchors: bool) -> list[str]:
    """The notes to print. When no column has anchors, the per-column 'only N anchor pairs'
    lines collapse into one line; every other note stays."""
    if not no_anchors:
        return list(notes)
    out: list[str] = []
    placed = False
    for t in notes:
        if _PER_COLUMN_ANCHOR_NOTE.search(t):
            if not placed:
                out.append(NO_ANCHORS_NOTE)
                placed = True
            continue
        out.append(t)
    return out if placed else [NO_ANCHORS_NOTE, *out]


def _report_reason(reason: str, no_anchors: bool) -> str:
    """With no anchors the marker in the Reliability cell says it; the reason keeps its facts."""
    return reason.split(". Reliability was not measured")[0] if no_anchors else reason


def _summary_paragraph(result: dict[str, Any], measured: bool) -> str:
    ordered = _ordered_columns(result)
    top = [f"{_md_cell(n)} ({c['composite_weight']:.3f})" for n, c in ordered[:3]]
    heaviest = (top[0] if len(top) == 1 else ", ".join(top[:-1]) + " and " + top[-1]) if top else "none"
    rel = ("Reliability was measured from pairs of records that likely refer to the same thing."
           if measured else
           "Reliability was not measured, so the weights come from distinctiveness only.")
    return (f"Verity analysed {result['n_records']:,} rows and kept {len(ordered)} column{'' if len(ordered) == 1 else 's'}. "
            f"The {'three ' if len(top) == 3 else ''}heaviest column{'s are' if len(top) > 1 else ' is'} "
            f"{heaviest}. {rel}")


def build_report(result: dict[str, Any], source: Path, n_original: int,
                 debug: dict[str, Any], seed: int, n_pairs_written: int,
                 pairs_written: bool, masked: set[str],
                 exclude: list[str] = (), keep: list[str] = (),
                 sampling: dict[str, Any] | None = None) -> str:
    cols = result["columns"]
    ordered = _ordered_columns(result)
    anchor_counts = debug.get("reliability_debug", {}).get("anchor_count")
    levels = debug.get("reliability_debug", {}).get("min_agree_used", {})
    n = result["n_records"]
    n_cand = result["run"]["n_candidate_pairs"]
    embeddings_used = result["embeddings_used"]
    L: list[str] = []
    measured = _reliability_measured(result, anchor_counts)
    no_anchors = _no_anchors(result, anchor_counts)
    L += [f"# Verity rules report: {source.name}", "",
          *(_read_first_box(result) if n_cand == 0 else []),
          _summary_paragraph(result, measured), "",
          "## What is Verity?", "",
          "Verity reads a dataset and learns which columns help identify the same real-world "
          "entity, and how much each column should count. It outputs rules (weights and "
          "comparison methods) and does not match records itself.", "",
          "## Dataset", "",
          f"- Source file: `{source.name}`",
          f"- Rows in file: {n_original:,}",
          f"- Rows analysed: {n:,}" + (" (a sample; see Limits)" if n < n_original else ""),
          *([f"- Sample mode: {sampling['mode']}"
             + (f"; groups of linked rows kept: {sampling['groups_kept']:,}"
                if sampling["mode"] == "groups" else "")]
            if sampling and n < n_original else []),
          f"- Record id column: `{result['record_id_column']}`", "",
          "## Run details", "",
          f"- Verity version: {result['run']['verity_version']}",
          f"- Seed: {seed}",
          f"- File encoding: {result['run']['encoding']}",
          f"- Delimiter: {result['run']['delimiter']}",
          f"- Run date (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
          f"- Contract version: {result['contract_version']}",
          f"- Embeddings used: {'yes' if embeddings_used else 'no'}",
          *([f"- Excluded by you (--exclude-columns): {', '.join(exclude)}"] if exclude else []),
          *([f"- Kept by you (--keep-columns): {', '.join(keep)}"] if keep else []),
          f"- Random pairs sampled: {result['n_pairs_sampled']:,}",
          f"- Candidate pairs found: {n_cand:,}",
          f"- Calibration pairs: {n_pairs_written}", "",
          "## Columns, by weight", "",
          "| Column | Weight | Reliability | Compared by | Reason |",
          "|---|---|---|---|---|"]
    any_low = any_unmeasured = False
    for name, c in ordered:
        anchors = None if anchor_counts is None else anchor_counts.get(name, 0)
        low = levels.get(name, ANCHOR_LEVEL) < ANCHOR_LEVEL
        unmeasured = anchors is not None and anchors < ANCHOR_MIN_COUNT
        any_unmeasured |= unmeasured
        any_low |= low and not unmeasured
        note = (" (not measured)" if unmeasured else " (low confidence)" if low else "")
        L.append(f"| {_md_cell(name)} | {c['composite_weight']:.3f} | "
                 f"{c['reliability_score']:.3f}{note} | "
                 f"{c['comparison_method']} | {_md_cell(_report_reason(c['reason'], no_anchors))} |")
    L += ["", "**Caution:** a weight is how distinctive the column is in this dataset, times its "
          "reliability, scaled so all weights sum to 1. A column that is generally a weak "
          "identifier can still score high here if it happens to be distinctive and "
          "consistent in this data. Read the weights together with the reasons.", "",
          "Compare these weights with what you know about the data before using them. A column "
          "that is generally a weak identifier can outrank a stronger one if its values are "
          "more distinct here.", "",
          "### Details", "",
          "| Column | Anchors |", "|---|---|"]
    for name, _ in ordered:
        L.append(f"| {_md_cell(name)} | "
                 f"{'n/a' if anchor_counts is None else anchor_counts.get(name, 0)} |")
    L += ["", "`columns.csv` holds every per-column field for opening in Excel.", "",
          "Candidate pairs are pairs of records that share an identical value in a column where "
          "at least half the values are distinct (and the value is shared by 2 to 10 records). "
          "Anchors are the pairs used to measure that column's reliability: candidate pairs "
          "that agree on at least two other columns, or on one when too few pairs qualify.", "",
          *(["**Low confidence** marks a column whose reliability was measured with the "
             "relaxed rule (one agreeing column). Treat its reliability and weight as less "
             "certain.", ""] if any_low else []),
          *(["**Not measured** marks a column with too few anchors to measure reliability; it "
             "is set to 1.0 and does not change the weight. Treat its reliability and weight "
             "as less certain.", ""] if any_unmeasured else []),
          "### Comparison methods", ""]
    methods = sorted({c["comparison_method"] for c in cols.values()})
    L += [f"- **{m}**: {_METHOD_GLOSSARY.get(m, 'See the Verity documentation.')}" for m in methods]
    if not embeddings_used:
        L += ["", "No embeddings were used in this run, so text columns are compared by adjacent character pairs (bigrams)."]
    L += ["", "## Excluded columns", ""]
    if result["columns_excluded"]:
        for e in result["columns_excluded"]:
            name = e["name"]
            if e["rule"] == [USER_EXCLUDED_RULE]:
                L.append(f"- `{name}`: Excluded by you.")
                continue
            if name == result["record_id_column"]:
                origin = "Verity rule."
            else:
                origin = f"Verity rule. To include it anyway, use `--keep-columns {name}`."
            L.append(f"- `{name}`: {e['reason']} ({origin})")
    else:
        L.append("No columns were excluded.")
    coverage = result["expected_pair_coverage"]
    L += ["", "## Expected pair coverage", "",
          f"**{coverage:.3f}**", "",
          "Expected pair coverage is the sum, over the kept columns, of the column's weight times "
          "the chance that both records of a pair have a value in it. That chance is (1 minus the "
          "column's null rate) squared, because both records need a value. It runs from 0 to 1. "
          "A high number means a typical pair can be compared on most of the evidence. A low "
          "number means many pairs will share only a few columns.", "",
          f"Rule of thumb: below about {COVERAGE_LOW}, many true duplicates will share too few "
          "columns to be decided, so send low-coverage pairs to review instead of deciding them "
          "automatically. This rule of thumb comes from synthetic tests, not from your data."]
    L += ["", "## Notes", ""]
    L += [f"- {_md_cell(t)}" for t in _report_notes(result["notes"], no_anchors)] or ["None."]
    L += ["", "## How to use these rules", "",
          "- **Weights** show how much each column should count. They sum to 1. They "
          "describe the data, so you can use them in any matching tool.",
          "- **Reliability** is how much more alike a column's values are across records that "
          "likely refer to the same thing than across random pairs of records. 0 means no better "
          "than chance and 1 means the values are always the same.",
          "- **Comparison method** says how to compare two values of that column. Each "
          "comparison gives a similarity from 0 (different) to 1 (the same). If your tool does "
          "not have this comparator, use its closest one: edit distance or Jaro-Winkler for "
          "text, exact match for ids. For email, compare the part before the @ and the domain "
          "separately if you can. The weights stay valid. Set the cutoff using the comparator "
          "you actually use, because a cutoff only makes sense for the scoring it was chosen "
          "with.", "",
          "### Pair score", "",
          "For two records, score = sum of (weight x similarity) over the columns that have a "
          "value in both records, divided by the sum of the weights of those same columns.", "",
          "Missing values: a column is skipped when either record has no value (blank, or the "
          "text nan, none, null or n/a). The skipped column leaves both the top and the bottom "
          "of the fraction, so the remaining weights are scaled up to fill its place. If no "
          "column has a value in both records, the score is 0. A pair compared on only a few "
          "columns can therefore score high on little evidence.", "",
          "### Pair coverage", "",
          "Coverage for a pair = the sum of the weights of the columns present in both records, "
          "divided by the sum of all weights. It is 1 when every column has a value in both "
          "records and falls as columns go missing. Pairs with low coverage are scored on little "
          "evidence.", "",
          f"On your labelled pairs, check whether low-coverage pairs are decided worse than the "
          f"rest (for example, compare the wrong-call rate for pairs below {COVERAGE_LOW} with "
          "the rate for the others). If they are, send pairs below the coverage level you choose "
          "to review instead of deciding them automatically.", "",
          "### Choosing a match cutoff", ""]
    no_matches = pairs_written and not result["calibration_pairs"]["likely_match"]
    if no_matches:
        L += ["This run found no likely matches, so calibration_pairs.csv cannot be used to "
              "choose a cutoff. Add known matches from your own data. To set a cutoff, label "
              "pairs from your own data and follow steps 2 to 6 below.", ""]
    else:
        L += ["Verity does not provide a cutoff. To set your own:", ""]
    if pairs_written and not no_matches:
        L += ["1. Open `calibration_pairs.csv` and fill in `your_label` for each pair: match "
              "or not a match. Label at least 30 pairs of each kind. If the file does not "
              "have 30 of a kind, add pairs from your own data. Leave "
              "`calibration_pairs_key.csv` unopened until you finish; it holds Verity's "
              "guesses and where each pair came from, which are not the truth. The `unclear` "
              "pairs mix mid-range candidate pairs with the highest-scoring pairs that are not "
              "candidates (near-misses)."]
    elif not no_matches:
        L += ["1. Take pairs from your own data and label each as a match or not a match. "
              "Label at least 30 pairs of each kind. (No pairs files were written because "
              "`--no-pairs` was used.)"]
    L += ["2. Score every labelled pair in your tool, using the formula above.",
          "3. Sort the pairs by score.",
          "4. For each candidate cutoff, count the wrong calls: matches that score below it "
          "plus non-matches that score at or above it.",
          "5. Pick the cutoff with the fewest wrong calls. If several cutoffs tie for the "
          "fewest, pick the middle of the tied range.",
          "6. If a false merge costs you more than a missed match, move the cutoff up."]
    if pairs_written:
        L += ["", "### Handling the pairs files", "",
              "`calibration_pairs.csv`, `calibration_pairs_key.csv` and the pairs inside "
              "`rules.json` contain values from the source data"
              + (f" (masked columns: {', '.join(sorted(masked))})" if masked else "")
              + ". Handle them like the source data."]
    L += ["", "## Limits", "",
          "- Verity does not match or clean records.",
          ("- The weights come from one run. " + (
              f"All {n:,} rows analysed were used; only pairs of records are sampled."
              if n >= n_original else
              f"{n:,} of the {n_original:,} rows were analysed (sample mode: "
              f"{(sampling or {}).get('mode', 'groups')}); within them, only pairs of records are sampled.")
           + " Rerun if the data changes."),
          *(["- Rows that share an identical value in a distinctive column were kept together when "
             "choosing the rows. Duplicates that share no identical value in a distinctive column "
             "can still be split by the sampling."]
            if n < n_original and (sampling or {}).get("mode") == "groups" else []),
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
    """Write the shuffled pairs file (no labels) and the key file (pair_id, suggested_label, source)."""
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
        kw.writerow(["pair_id", "suggested_label", "source"])
        for k, (cat, pair) in enumerate(flat, start=1):
            row = [k, pair["record_id_a"], pair["record_id_b"]]
            for c in content:
                row += [pair["values_a"].get(c), pair["values_b"].get(c)]
            row.append("")
            w.writerow(["" if v is None or v != v else v for v in row])  # v != v: NaN
            kw.writerow([k, cat, pair["source"]])
    return len(flat)


def write_columns_csv(path: Path, result: dict[str, Any],
                      anchor_counts: dict[str, int] | None) -> None:
    """One row per kept column with every per-column field, heaviest first. UTF-8 with a BOM
    so Excel reads accents correctly."""
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["column", *COLUMN_FIELDS, "anchors"])
        for name, c in _ordered_columns(result):
            w.writerow([name, *(c[f] for f in COLUMN_FIELDS),
                        "" if anchor_counts is None else anchor_counts.get(name, 0)])


def _name_list(text: str | None) -> list[str]:
    return [c.strip() for c in text.split(",") if c.strip()] if text else []


def write_outputs(out: Path, src: Path, result: dict[str, Any], debug: dict[str, Any],
                  args: argparse.Namespace, n_original: int, mask: set[str], exclude: list[str],
                  keep: list[str], sampling: dict[str, Any]) -> tuple[list[str], int]:
    """Write the output files. Returns (file names written, number of pairs written)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "rules.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_columns_csv(out / "columns.csv", result, debug.get("reliability_debug", {}).get("anchor_count"))
    written = ["rules.json", "rules_report.md", "columns.csv"]
    n_pairs = 0
    if not args.no_pairs:
        n_pairs = write_calibration_csv(out / "calibration_pairs.csv",
                                        out / "calibration_pairs_key.csv", result, args.seed)
        written += ["calibration_pairs.csv", "calibration_pairs_key.csv"]
    (out / "rules_report.md").write_text(
        build_report(result, src, n_original, debug, args.seed, n_pairs,
                     not args.no_pairs, mask, exclude, keep, sampling), encoding="utf-8")
    if args.debug:
        (out / "debug.json").write_text(json.dumps(debug, indent=2, allow_nan=False), encoding="utf-8")
        written.append("debug.json")
    return written, n_pairs


def cmd_learn(args: argparse.Namespace) -> int:
    src, out = Path(args.csv), Path(args.out)
    if args.max_rows < 2:
        raise CliError("--max-rows must be at least 2.")
    existing = [f for f in (*OUTPUT_FILES, *(["debug.json"] if args.debug else []))
                if (out / f).exists()]
    if existing and not args.overwrite:
        raise CliError(f"{', '.join(existing)} already exist{'s' if len(existing) == 1 else ''} in "
                       f"{out}; use --overwrite to replace {'it' if len(existing) == 1 else 'them'}, "
                       "or choose another --out.")
    exclude, keep = _name_list(args.exclude_columns), _name_list(args.keep_columns)
    both = sorted(set(exclude) & set(keep))
    if both:
        raise CliError("--exclude-columns and --keep-columns both list: " + ", ".join(both))
    records, file_info = read_table(src, args.encoding, args.delimiter)
    if args.encoding is None and file_info["encoding"] == "cp1252":
        print("Note: the file is not valid UTF-8, so it was read as cp1252. "
              "Use --encoding to choose another.", file=sys.stderr)
    records, id_column, row_numbers = choose_record_id(records, args.id_column)
    row_note = (f"No unique id column was found. Row numbers ({id_column}) are used as record ids "
                "in the pairs files.")
    if row_numbers:
        print(f"Warning: {row_note}", file=sys.stderr)
    guess_note = None
    if not row_numbers and args.id_column is None and not is_idlike_name(id_column):
        guess_note = (f"Using {id_column} as the record id because it has no blanks and almost every "
                      "value is distinct. If this is not an id, use --id-column or add row numbers.")
        print(f"Warning: {guess_note}", file=sys.stderr)
    records, sampling = sample_records(records, args.max_rows, args.seed, args.sample_mode,
                                       exclude, keep, id_column)
    n_original = sampling["rows_in_file"]
    try:
        result = learn_rules(records, seed=args.seed, debug=True,
                             exclude_columns=exclude, keep_columns=keep,
                             record_id_column=id_column)
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith(("exclude_columns", "keep_columns")):
            raise CliError(msg.replace("exclude_columns", "--exclude-columns")
                           .replace("keep_columns", "--keep-columns")) from None
        raise CliError(str(exc)) from None

    debug = {k: result.pop(k) for k in DEBUG_KEYS if k in result}
    if n_original > len(records):
        result["notes"].append(
            f"Dataset has {n_original:,} rows; {len(records):,} were analysed "
            f"(sample mode {sampling['mode']}, seed {args.seed}).")
    if row_numbers:
        result["notes"].append(row_note)
    if guess_note:
        result["notes"].append(guess_note)
    result["run"]["sampling"] = sampling
    result["run"].update(file_info)
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

    try:
        written, n_pairs = write_outputs(out, src, result, debug, args, n_original, mask,
                                         exclude, keep, sampling)
    except OSError as exc:
        target = exc.filename or out
        raise CliError(f"cannot write {target} ({exc.strerror or exc}). If it is open in Excel "
                       "or another program, close it and run again.") from None
    if result["run"]["n_candidate_pairs"] == 0:
        print("Warning: " + NO_CANDIDATES_NOTE, file=sys.stderr)
    print(f"Learned rules for {len(records):,} rows, {len(result['columns'])} columns, "
          f"{n_pairs} calibration pairs.")
    print(f"Wrote {', '.join(written)} to {out}")
    if not args.no_pairs:
        print(PAIRS_NOTICE)
    return 0


MANY_ROWS_MEDIAN = 10
MAY_NOT_SUIT = ("Verity may not suit this file. This usually means each entity appears in many "
                "rows, or duplicates differ in every column.")


def _group_sizes(values: list[Any]) -> list[int]:
    """Sizes of the groups of repeated values (2 or more rows), grouped as candidate pairs do."""
    counts: dict[str, int] = {}
    for v in values:
        if v is not None and v == v and str(v).strip():
            key = str(v).strip().lower()
            counts[key] = counts.get(key, 0) + 1
    return [c for c in counts.values() if c >= 2]


def check_file(records: list[dict[str, Any]], id_column: str) -> dict[str, Any]:
    """What a learn run would find in this file, without learning rules. Uses the same column
    classification, profiles and candidate pairs as learn, and the same anchor counting."""
    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)
    n = len(df)
    cls = classify_columns(df, id_column)
    rows = df.to_dict(orient="records")
    content = [h for h in df.columns if h in set(cls["content_columns"])]
    profiles = build_profiles(content, rows, set(cls["metadata_columns"]))
    stats = {}
    for c in df.columns:
        uniq, blank = column_uniqueness_and_nulls(df[c].tolist(), n)
        stats[c] = {"distinct": uniq, "blank": blank}
    per_col = {p.name: len(_build_candidates(rows, [p])) for p in profiles
               if p.uniqueness_ratio >= CANDIDATE_MIN_UNIQUENESS}
    cand_pairs = sorted(_build_candidates(rows, profiles))
    n_cand = len(cand_pairs)
    capped = n_cand > CANDIDATE_CAP
    if capped:
        cand_pairs = sorted(random.Random(DEFAULT_SEED).sample(cand_pairs, CANDIDATE_CAP))
    anchors: dict[str, int] = {}
    if cand_pairs:
        lo_hi = {p.name: numeric_series_stats(p.values) for p in profiles}
        _, info = learn_reliability_v2(cand_pairs, rows, profiles, lo_hi)
        anchors = info["anchor_count"]
    measured = [p.name for p in profiles if anchors.get(p.name, 0) >= ANCHOR_MIN_COUNT]
    top = max(profiles, key=lambda p: (p.uniqueness_ratio, p.name), default=None)
    sizes = _group_sizes(top.values) if top else []
    return {"n_rows": n, "stats": stats,
            "excluded": sorted(set(cls["metadata_columns"]) - {id_column}),
            "candidate_columns": [c for c, k in per_col.items() if k > 0],
            "any_distinct": bool(per_col), "n_candidate_pairs": n_cand, "capped": capped,
            "measured": measured,
            "unmeasured": [p.name for p in profiles if p.name not in measured],
            "median_group": statistics.median(sizes) if sizes else 0,
            "most_distinct": top.name if top else None}


def check_outcome(r: dict[str, Any]) -> str:
    """One plain line: whether reliability can be measured on this file."""
    if r["n_candidate_pairs"] == 0:
        why = ("no column has at least half of its values distinct" if not r["any_distinct"] else
               f"no value in a column with at least half of its values distinct is shared by "
               f"{CANDIDATE_GROUP_MIN} to {CANDIDATE_GROUP_MAX} rows")
        return f"Expected outcome: reliability would not be measured, because {why}."
    if not r["measured"]:
        return ("Expected outcome: reliability would not be measured, because too few candidate "
                f"pairs agree on other columns (each column needs {ANCHOR_MIN_COUNT} such pairs).")
    if r["unmeasured"]:
        return ("Expected outcome: reliability can be measured only partly: for "
                f"{', '.join(r['measured'])}; not for {', '.join(r['unmeasured'])} "
                "(too few anchor pairs).")
    return "Expected outcome: reliability can be measured for every column."


def cmd_check(args: argparse.Namespace) -> int:
    """Print what a learn run would find. Writes nothing."""
    src = Path(args.csv)
    records, file_info = read_table(src, args.encoding, args.delimiter)
    if args.encoding is None and file_info["encoding"] == "cp1252":
        print("Note: the file is not valid UTF-8, so it was read as cp1252. "
              "Use --encoding to choose another.", file=sys.stderr)
    records, id_column, row_numbers = choose_record_id(records, None)
    r = check_file(records, id_column)
    if row_numbers:
        id_line = "row numbers (no unique id column was found)"
    else:
        id_line = id_column + ("" if is_idlike_name(id_column) else " (guessed from the data)")
    L = [f"File: {src.name} ({file_info['delimiter']}-delimited, {file_info['encoding']})",
         f"Rows: {r['n_rows']:,}", f"Record id: {id_line}", "",
         "Per column (distinct = share of non-blank values that are different):"]
    shown = [c for c in r["stats"] if not (row_numbers and c == id_column)]
    width = max(len(c) for c in shown)
    L.append(f"  {'column':<{width}}  distinct  blank")
    for c in shown:
        st = r["stats"][c]
        L.append(f"  {c:<{width}}  {st['distinct']:>7.0%}  {st['blank']:>5.0%}")
    if r["excluded"]:
        L.append(f"Verity would exclude: {', '.join(r['excluded'])}")
    L += ["", "Columns that could produce candidate pairs (at least half distinct, values shared "
          f"by {CANDIDATE_GROUP_MIN} to {CANDIDATE_GROUP_MAX} rows): "
          + (", ".join(r["candidate_columns"]) or "none"),
          f"Candidate pairs: {r['n_candidate_pairs']:,}"
          + (f" (learn would use {CANDIDATE_CAP:,} of them)" if r["capped"] else ""),
          check_outcome(r)]
    if r["n_candidate_pairs"] == 0 or not r["measured"] or r["unmeasured"]:
        L.append(MAY_NOT_SUIT)
    if r["median_group"] > MANY_ROWS_MEDIAN:
        L.append(f"Note: the most distinctive column ({r['most_distinct']}) repeats its values in "
                 f"groups of {r['median_group']:g} rows (median). This looks like a table with "
                 "many rows per entity, so Verity may not suit it.")
    if r["n_rows"] > DEFAULT_MAX_ROWS:
        L.append(f"Note: counts are for the whole file; learn analyses at most "
                 f"{DEFAULT_MAX_ROWS:,} rows by default (--max-rows).")
    print("\n".join(L))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verity", description="Learn entity-matching rules from a CSV.")
    p.add_argument("--version", action="version", version=f"verity {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    learn = sub.add_parser("learn", help="learn column weights and rules from a CSV")
    learn.add_argument("csv", help="path to the input CSV")
    learn.add_argument("--out", required=True, help="output folder (created if missing)")
    learn.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default 42)")
    learn.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                       help="analyse at most this many rows if the file is larger (default 20000)")
    learn.add_argument("--sample-mode", choices=SAMPLE_MODES, default="groups",
                       help="how rows are chosen above --max-rows: groups keeps rows that share a "
                            "distinctive value together, random draws rows uniformly (default groups)")
    learn.add_argument("--mask-columns", metavar="COLS",
                       help="comma-separated column names; their values become *** in the pairs "
                            "(calibration_pairs.csv and rules.json)")
    learn.add_argument("--exclude-columns", metavar="COLS",
                       help="comma-separated column names to leave out even if Verity would use them")
    learn.add_argument("--keep-columns", metavar="COLS",
                       help="comma-separated column names to use even if a Verity rule would "
                            "exclude them (not the record id column)")
    learn.add_argument("--id-column", metavar="NAME",
                       help="use this column as the record id (no blank values, at least 99%% "
                            "distinct values). Default: Verity picks one, or uses row numbers")
    learn.add_argument("--overwrite", action="store_true",
                       help="replace output files that already exist in --out (default: refuse)")
    check = sub.add_parser("check", help="show what a learn run would find in a CSV; writes nothing")
    check.add_argument("csv", help="path to the input CSV")
    for sp in (learn, check):
        sp.add_argument("--encoding", metavar="NAME",
                        help="text encoding of the CSV, such as utf-8 or cp1252. Default: utf-8, "
                             "then cp1252 if the file is not valid UTF-8")
        sp.add_argument("--delimiter", metavar="CHAR",
                        help="field delimiter: comma, semicolon, tab, pipe or one character. "
                             "Default: detected from the first 20 lines")
    check.set_defaults(func=cmd_check)
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
