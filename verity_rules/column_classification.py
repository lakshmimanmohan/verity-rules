"""
Data-driven column metadata vs content classification for arbitrary CSVs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLUMN_CLASSIFICATION_GLOBAL_NAME = "column_classification_global.json"


def global_column_classification_path() -> Path:
    """``data/column_classification_global.json`` under the project root."""
    return PROJECT_ROOT / "data" / COLUMN_CLASSIFICATION_GLOBAL_NAME


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

# Column names always metadata regardless of value patterns (belt-and-suspenders).
ALWAYS_METADATA_NAMES = frozenset({"source_system"})
_ALWAYS_METADATA_RE = re.compile(
    r"^(source[_\s]?system|source)$",
    re.IGNORECASE,
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


def pick_record_id_column(df: pd.DataFrame, primary_key_columns: list[str]) -> str:
    """Prefer canonical id column names, then PK-classified columns, then highest cardinality."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    id_suffix_re = re.compile(r"(_id|_key)$", re.IGNORECASE)
    dynamic_prefs = [kl for kl in lower_map if id_suffix_re.search(kl)]
    static_prefs = ("id", "record_id", "uuid", "key")
    prefs = (*dynamic_prefs, *static_prefs)
    for p in prefs:
        if p in lower_map:
            return lower_map[p]
    pk_lower = {str(c).strip().lower(): c for c in primary_key_columns}
    for p in prefs:
        if p in pk_lower:
            return pk_lower[p]
    if primary_key_columns:
        order = {c: i for i, c in enumerate(df.columns)}
        return sorted(primary_key_columns, key=lambda c: order.get(c, 999))[0]
    best, best_u = df.columns[0], -1
    for c in df.columns:
        u = int(df[c].map(token_key).nunique())
        if u > best_u:
            best, best_u = c, u
    return str(best)


def classify_columns(df: pd.DataFrame) -> dict[str, Any]:
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
         and uniqueness ratio < 5%); or reserved / always-metadata name pattern.

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

        col_lower = str(col).strip().lower()
        if _ALWAYS_METADATA_RE.match(col_lower):
            rule_hits.append(
                (
                    "always_metadata_name",
                    f"column name {col_lower!r} is always metadata (system lineage / provenance)",
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

    record_id_column = pick_record_id_column(df, primary_key_columns)

    metadata_sorted = sorted(metadata)
    content_sorted = sorted(c for c in content if c not in metadata)
    if record_id_column in content_sorted:
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


def print_classification_report(data: dict[str, Any]) -> None:
    print("=" * 72)
    print("  Column classification (metadata vs content)")
    print("=" * 72)
    meta = data.get("metadata_columns") or []
    cont = data.get("content_columns") or []
    rid = data.get("record_id_column", "")
    print(f"  record_id_column: {rid!r}")
    print(f"  metadata_columns ({len(meta)}): {', '.join(meta) or '(none)'}")
    print(f"  content_columns ({len(cont)}): {', '.join(cont) or '(none)'}")
    print("-" * 72)
    print("  Per-column reasoning:")
    for col, why in sorted((data.get("reasoning") or {}).items()):
        print(f"    {col}: {why}")
    print("=" * 72 + "\n")


def write_column_classification(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_column_classification(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _non_empty_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(c).strip() for c in raw if str(c).strip()]


def merge_global_column_classification_snapshot(
    previous: dict[str, Any] | None,
    file_scoped: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the aggregate document for ``column_classification_global.json``.
    ``content_columns`` and ``metadata_columns`` are set unions across snapshots;
    ``semantic_types`` / ``reasoning`` merge with the latest file-scoped write winning
    on duplicate keys; ``record_id_column`` prefers the current file.
    """
    prev_content = set(_non_empty_str_list((previous or {}).get("content_columns")))
    file_content = set(_non_empty_str_list(file_scoped.get("content_columns")))
    union_content = sorted(prev_content | file_content, key=str.lower)

    prev_meta = set(_non_empty_str_list((previous or {}).get("metadata_columns")))
    file_meta = set(_non_empty_str_list(file_scoped.get("metadata_columns")))
    union_meta = sorted(prev_meta | file_meta, key=str.lower)

    merged_st: dict[str, Any] = {}
    for src in ((previous or {}).get("semantic_types"), file_scoped.get("semantic_types")):
        if not isinstance(src, dict):
            continue
        for k, v in src.items():
            sk = str(k).strip()
            if sk:
                merged_st[sk] = v

    merged_reason: dict[str, Any] = {}
    prev_r = (previous or {}).get("reasoning")
    if isinstance(prev_r, dict):
        merged_reason.update(prev_r)
    file_r = file_scoped.get("reasoning")
    if isinstance(file_r, dict):
        merged_reason.update(file_r)

    rid = str(file_scoped.get("record_id_column") or "").strip()
    if not rid and previous:
        rid = str((previous or {}).get("record_id_column") or "").strip()

    return {
        "content_columns": union_content,
        "metadata_columns": union_meta,
        "record_id_column": rid,
        "reasoning": merged_reason,
        "semantic_types": merged_st,
    }


def write_file_scoped_column_classification(path: Path, data: dict[str, Any]) -> None:
    """
    Write file-scoped ``column_classification.json`` and refresh
    ``data/column_classification_global.json`` by unioning ``content_columns`` (and
    related fields) with any existing global snapshot.
    """
    write_column_classification(path, data)
    global_path = global_column_classification_path()
    previous = load_column_classification(global_path)
    merged = merge_global_column_classification_snapshot(previous, data)
    write_column_classification(global_path, merged)


def apply_global_content_overlay(base: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of ``base`` with ``content_columns`` unioned against
    ``data/column_classification_global.json`` when that file exists.
    ``semantic_types`` are merged so file-scoped entries override global;
    global supplies types for columns only seen on other files.
    """
    g = load_column_classification(global_column_classification_path())
    if g is None:
        return dict(base)
    out = dict(base)
    file_content = set(_non_empty_str_list(out.get("content_columns")))
    glob_content = set(_non_empty_str_list(g.get("content_columns")))
    out["content_columns"] = sorted(file_content | glob_content, key=str.lower)

    merged_st: dict[str, Any] = {}
    g_st = g.get("semantic_types")
    if isinstance(g_st, dict):
        for k, v in g_st.items():
            sk = str(k).strip()
            if sk:
                merged_st[sk] = v
    b_st = out.get("semantic_types")
    if isinstance(b_st, dict):
        for k, v in b_st.items():
            sk = str(k).strip()
            if sk:
                merged_st[sk] = v
    out["semantic_types"] = merged_st
    return out


def classification_or_infer(csv_path: Path) -> dict[str, Any]:
    """Load ``column_classification.json`` next to CSV, or build it with pandas."""
    p = csv_path.parent / "column_classification.json"
    loaded = load_column_classification(p)
    if loaded is not None:
        return loaded
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    data = classify_columns(df)
    write_file_scoped_column_classification(p, data)
    return data


def column_to_attr_type(col: str) -> str:
    """Neo4j AttributeValue.type — uppercase snake."""
    assert str(col).strip().lower() != "source_system", (
        "source_system must not be passed to column_to_attr_type; "
        "it is modeled on CustomerRecord and SourceSystem, not AttributeValue"
    )
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").upper()
    return s or "ATTRIBUTE"


def neo_property_key(col: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
