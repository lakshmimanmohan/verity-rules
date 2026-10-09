"""
Data-driven column metadata vs content classification for arbitrary CSVs.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


EMAILISH_RE = re.compile(r"^[^@\s]+@.+\..+$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z\s\-'.]*$")


def is_null(v: Any) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "n/a"}


def clean_str(v: Any) -> str:
    return str(v).strip() if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""


def token_key(v: Any) -> str:
    if is_null(v):
        return "__NULL__"
    return clean_str(v).lower()


def proportion_human_text(values: list[Any]) -> float:
    """Share of non-null values that look like names, emails, or multi-word text."""
    nn = [clean_str(v) for v in values if not is_null(v)]
    if not nn:
        return 0.0
    ok = 0
    for s in nn:
        low = s.lower()
        if EMAILISH_RE.match(low.replace("(at)", "@")):
            ok += 1
            continue
        if len(s) >= 8 and " " in s and re.search(r"[a-zA-Z]", s):
            ok += 1
            continue
        if NAME_RE.match(s) and re.search(r"[a-zA-Z]", s):
            ok += 1
            continue
    return ok / len(nn)


def is_reserved_metadata_name(col: str) -> bool:
    n = str(col).strip().lower()
    return (
        n.startswith("_")
        or n.endswith("_truth")
        or n.endswith("_gold")
        or n in {"label", "labels", "y", "target", "gold_label"}
    )


# System-process substrings in column *names* (always trigger when matched, except id/key/code).
_SYSTEM_NAME_ALWAYS = frozenset(
    {
        "source",
        "system",
        "status",
        "flag",
        "type",
        "tier",
        "segment",
        "class",
        "category",
        "created",
        "updated",
        "timestamp",
        "date",
        "modified",
        "score",
        "rank",
    }
)
# These name tokens only count as system metadata if the column is 100% unique (all rows distinct).
_SYSTEM_NAME_IF_UNIQUE = frozenset({"id", "key", "code"})
# Substring tokens that often appear inside legitimate column names unless bounded.
_SYSTEM_NAME_BOUNDARY_TOKENS = frozenset(
    {
        "class",
        "type",
        "tier",
        "rank",
        "date",
        "flag",
        "source",
        "system",
        "status",
        "segment",
    }
)

# Synthetic label value patterns (answer key).
_WORD_NUMBER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_\d+$")


def _name_has_process_token(name_lower: str, token: str) -> bool:
    """Match process-field token in column name; ambiguous short tokens use token boundaries."""
    if token in _SYSTEM_NAME_BOUNDARY_TOKENS:
        return bool(
            re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", name_lower)
        )
    return token in name_lower


def _name_has_id_key_code_token(name_lower: str, token: str) -> bool:
    """Avoid matching ``id`` inside ``middle`` — token as whole segment (^_ or _tok_ or _tok$)."""
    return bool(re.search(rf"(^|_){re.escape(token)}(_|$)", name_lower))


def is_system_process_field_name(
    col: str,
    uniq_ratio: float,
    n_rows: int,
    n_distinct: int,
    n_non_null: int,
    n_distinct_nn: int,
) -> tuple[bool, str]:
    """
    Rule 2: name hints at lifecycle / process, not entity.
    ``id`` / ``key`` / ``code`` in the name only when values are 100% unique (all rows non-null, all distinct).
    """
    if n_rows <= 0:
        return False, ""
    name_l = str(col).strip().lower()
    full_unique = (
        n_non_null == n_rows
        and n_distinct_nn == n_non_null
        and n_distinct == n_rows
    )
    hits_always: list[str] = []
    hits_cond: list[str] = []
    for tok in sorted(_SYSTEM_NAME_ALWAYS):
        if _name_has_process_token(name_l, tok):
            hits_always.append(tok)
    for tok in sorted(_SYSTEM_NAME_IF_UNIQUE):
        if _name_has_id_key_code_token(name_l, tok):
            hits_cond.append(tok)
    if hits_always:
        return True, "system process field (name contains: " + ", ".join(hits_always) + ")"
    if hits_cond and full_unique:
        return True, (
            "system process field (name contains: "
            + ", ".join(hits_cond)
            + "; only flagged because values are 100% unique)"
        )
    return False, ""


def is_answer_key_values(values: list[Any], min_fraction: float = 0.72) -> tuple[bool, str]:
    """
    Rule 3: synthetic labels — WORD_NUMBER or UNKNOWN_ prefix on most non-null values.
    """
    nn = [clean_str(v) for v in values if not is_null(v)]
    if len(nn) < 3:
        return False, ""
    ok = 0
    for s in nn:
        t = s.strip()
        if not t:
            continue
        if t.upper().startswith("UNKNOWN_"):
            ok += 1
            continue
        if _WORD_NUMBER_RE.match(t):
            ok += 1
            continue
    frac = ok / len(nn)
    if frac >= min_fraction:
        return True, (
            f"answer key (≥{min_fraction:.0%} of values match WORD_NUMBER or UNKNOWN_* pattern)"
        )
    return False, ""


def is_boolean_flag_column(col: str, n_distinct: int) -> tuple[bool, str]:
    """Rule 4: 2–3 distinct values and name suggests a boolean flag."""
    if n_distinct < 2 or n_distinct > 3:
        return False, ""
    name_l = str(col).strip().lower()
    flag_hints = ("is_", "has_", "active", "enabled", "verified", "opted")
    if not any(h in name_l for h in flag_hints):
        return False, ""
    return True, (
        "boolean/binary flag (2–3 distinct values; name suggests is_/has_/active/enabled/verified/opted)"
    )


RECORD_ID_MIN_DISTINCT = 0.99


def record_id_problem(values: list[Any]) -> str | None:
    """Why a column cannot be a record id, or None if it can: it needs no blank values and at
    least 99 percent distinct values."""
    n = len(values)
    blanks = sum(1 for v in values if is_null(v) or not str(v).strip())
    if blanks:
        return f"it has {blanks:,} blank value{'s' if blanks != 1 else ''}"
    distinct = len({str(v).strip() for v in values})
    if n == 0 or distinct / n < RECORD_ID_MIN_DISTINCT:
        return f"only {distinct / max(n, 1):.1%} of its values are distinct (needs at least 99%)"
    return None


def pick_record_id_column(df: pd.DataFrame, primary_key_columns: list[str] | None = None) -> str | None:
    """Pick a record id among the columns that can be one (see record_id_problem): prefer
    canonical id column names, then the first such column in file order. primary_key_columns is
    kept for callers; every column that qualifies is already a near-unique key. Returns None when no column qualifies."""
    ok = [c for c in df.columns if record_id_problem(df[c].tolist()) is None]
    if not ok:
        return None
    lower_map = {str(c).strip().lower(): c for c in ok}
    id_suffix_re = re.compile(r"(_id|_key)$", re.IGNORECASE)
    dynamic_prefs = [kl for kl in lower_map if id_suffix_re.search(kl)]
    static_prefs = ("id", "record_id", "uuid", "key")
    prefs = (*dynamic_prefs, *static_prefs)
    for p in prefs:
        if p in lower_map:
            return lower_map[p]
    return str(ok[0])


def classify_columns(df: pd.DataFrame, record_id_column: str | None = None) -> dict[str, Any]:
    """
    Classify each column as metadata (excluded from matching) vs content.

    Metadata if **any** of:
      1) Primary key — uniqueness ratio ≥ 99%;
      2) System process field — name tokens (source, system, status, …);
         ``id`` / ``key`` / ``code`` in the name only when values are 100% unique;
      3) Answer key — most values look like WORD_NUMBER or UNKNOWN_*;
      4) Boolean flag — 2–3 distinct values and name suggests is_/has_/…;
      5) Degenerate field — exactly one distinct value (no discriminative power);
      6) Near-constant field — fewer than 5 distinct values, **or** (fewer than 10 distinct
         and uniqueness ratio < 5%); or reserved name pattern.

    Rules 5–6 are evaluated with other metadata rules before the human-readable text
    content signal, so they take priority over ``proportion_human_text``.

    Low-cardinality geographic / categorical fields (e.g. state, country) stay **content**
    unless another rule fires.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    n_rows = len(df)
    reasoning: dict[str, str] = {}
    metadata: set[str] = set()
    content: set[str] = set()
    primary_key_columns: list[str] = []

    for col in df.columns:
        series = df[col]
        values = series.tolist()
        keys = [token_key(v) for v in values]
        n_distinct = len(set(keys))
        n_non_null = sum(1 for v in values if not is_null(v))
        uniq_ratio = n_distinct / n_rows if n_rows else 0.0

        rule_hits: list[tuple[str, str]] = []

        if is_reserved_metadata_name(col):
            rule_hits.append(
                (
                    "reserved_name",
                    "reserved column name (_prefix, _truth/_gold, label/target/y, …)",
                )
            )

        if n_rows > 0 and uniq_ratio >= 0.99:
            rule_hits.append(
                (
                    "primary_key",
                    f"primary key (uniqueness ratio {uniq_ratio * 100:.2f}% ≥ 99%)",
                )
            )
            primary_key_columns.append(col)

        n_distinct_nn = len({token_key(v) for v in values if not is_null(v)})
        sys_ok, sys_detail = is_system_process_field_name(
            col, uniq_ratio, n_rows, n_distinct, n_non_null, n_distinct_nn
        )
        if sys_ok and sys_detail:
            rule_hits.append(("system_process_field", sys_detail))

        ak_ok, ak_detail = is_answer_key_values(values)
        if ak_ok and ak_detail:
            rule_hits.append(("answer_key", ak_detail))

        bf_ok, bf_detail = is_boolean_flag_column(col, n_distinct)
        if bf_ok and bf_detail:
            rule_hits.append(("boolean_flag", bf_detail))

        # RULE B — single value / degenerate (before human-readable content signal).
        if n_rows > 0 and n_distinct == 1:
            rule_hits.append(
                (
                    "degenerate_field",
                    "degenerate field, only 1 distinct value across all records — no discriminative power",
                )
            )
        # RULE A — near-constant: <5 distinct always, or <10 distinct with very low uniqueness.
        elif n_rows > 0 and (
            n_distinct < 5 or (n_distinct < 10 and uniq_ratio < 0.05)
        ):
            if n_distinct < 5:
                nc_detail = (
                    f"near-constant field, {n_distinct} distinct values (<5) — "
                    "system label / category regardless of dataset size"
                )
            else:
                nc_detail = (
                    f"near-constant field, {n_distinct} distinct values and "
                    f"uniqueness ratio {uniq_ratio:.4f} (<10 distinct and <5%)"
                )
            rule_hits.append(("near_constant_field", nc_detail))

        if rule_hits:
            metadata.add(col)
            parts = [f"[{rid}] {txt}" for rid, txt in rule_hits]
            reasoning[col] = "excluded: " + " | ".join(parts)
            continue

        hint_u = f"{uniq_ratio * 100:.1f}% unique, {n_distinct} distinct"
        if proportion_human_text(values) >= 0.35:
            reasoning[col] = f"included: {hint_u}; human-readable text signal"
        else:
            reasoning[col] = f"included: {hint_u}; no metadata rule matched (entity-describing field)"
        content.add(col)

    if record_id_column is None:
        record_id_column = pick_record_id_column(df, primary_key_columns)
    elif record_id_column not in df.columns:
        raise ValueError(f"record_id_column: no such column: {record_id_column}")

    metadata_sorted = sorted(metadata)
    content_sorted = sorted(c for c in content if c not in metadata)
    if record_id_column is not None and record_id_column in content_sorted:
        content_sorted = [c for c in content_sorted if c != record_id_column]
        metadata_sorted = sorted(set(metadata_sorted) | {record_id_column})
        extra = "[record_id_column] designated record id (removed from content_columns)"
        prev = reasoning.get(record_id_column, "")
        reasoning[record_id_column] = f"{prev} | {extra}" if prev else f"excluded: {extra}"

    out: dict[str, Any] = {
        "content_columns": content_sorted,
        "metadata_columns": sorted(set(metadata_sorted)),
        "record_id_column": record_id_column,
        "reasoning": {k: reasoning[k] for k in sorted(reasoning.keys())},
    }
    return out
