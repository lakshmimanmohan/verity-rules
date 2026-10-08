"""Group-aware row sampling. Pure: takes records, returns row indices. No file access.

When a file has more rows than the cap, random rows would split most duplicate sets apart.
Instead, rows that share a value in a distinctive column are kept together as a group.
"""
from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .column_classification import classify_columns
from .rule_learner import column_uniqueness_and_nulls, is_null

SOURCE_MIN_UNIQUENESS = 0.5  # same rule as core.CANDIDATE_MIN_UNIQUENESS
LINK_GROUP_MIN, LINK_GROUP_MAX = 2, 10  # same as core.CANDIDATE_GROUP_MIN / _MAX
MAX_GROUP_ROWS = 50  # larger groups are treated as single rows


def _normalize(v: Any) -> str:
    return "".join(ch for ch in str(v).lower() if ch.isalnum())


def source_columns(df: pd.DataFrame, exclude: list[str] = (), keep: list[str] = ()) -> list[str]:
    """Non-metadata columns whose distinct non-null values / non-null count is at least 0.5.
    exclude and keep are the user's lists: excluded columns never count, and kept columns are
    not dropped as metadata (the record id column stays out), as in learn_rules."""
    cls = classify_columns(df)
    metadata = (set(cls["metadata_columns"]) | set(exclude)) - (set(keep) - {cls["record_id_column"]})
    n = len(df)
    return [c for c in df.columns if c not in metadata
            and column_uniqueness_and_nulls(df[c].tolist(), n)[0] >= SOURCE_MIN_UNIQUENESS]


def build_groups(records: list[dict[str, Any]], columns: list[str]) -> list[list[int]]:
    """Connected components of rows linked by a normalized value shared by 2 to 10 rows in any of
    the columns. Groups are ordered by their first row; rows inside a group are ascending."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in columns:
        by_value: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            v = r.get(c)
            if is_null(v):
                continue
            key = _normalize(v)
            if key:
                by_value.setdefault(key, []).append(i)
        for members in by_value.values():
            if LINK_GROUP_MIN <= len(members) <= LINK_GROUP_MAX:
                root = find(members[0])
                for m in members[1:]:
                    parent[find(m)] = root
    comps: dict[int, list[int]] = {}
    for i in range(len(records)):
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=lambda g: g[0])


def group_sample_indices(records: list[dict[str, Any]], max_rows: int, seed: int = 42,
                         exclude: list[str] = (), keep: list[str] = ()) -> tuple[list[int], dict[str, Any]]:
    """Row indices (ascending) of at most max_rows rows, keeping linked rows together.

    Groups over MAX_GROUP_ROWS rows count as single rows. Groups (single rows included) are
    shuffled with the seed and added whole until the next one would exceed max_rows; the rest is
    filled with single rows. Returns (indices, info); info has the number of rows linked into
    groups, groups of 2+ rows kept, and groups treated as single rows because they were too big.
    exclude and keep are the final column lists (see source_columns)."""
    n = len(records)
    if n <= max_rows:
        return list(range(n)), {"groups_kept": 0, "oversized_groups": 0, "source_columns": []}
    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]
    cols = source_columns(df.where(pd.notna(df), None), exclude, keep)
    groups = build_groups(records, cols)
    oversized = [g for g in groups if len(g) > MAX_GROUP_ROWS]
    blocks = [g for g in groups if len(g) <= MAX_GROUP_ROWS] + [[i] for g in oversized for i in g]
    blocks.sort(key=lambda g: g[0])
    random.Random(seed).shuffle(blocks)
    keep: list[int] = []
    kept_groups = 0
    pos = 0
    while pos < len(blocks) and len(keep) + len(blocks[pos]) <= max_rows:
        keep += blocks[pos]
        kept_groups += len(blocks[pos]) > 1
        pos += 1
    if len(keep) < max_rows:
        for b in blocks[pos:]:
            if len(b) == 1:
                keep.append(b[0])
                if len(keep) == max_rows:
                    break
    return sorted(keep), {"groups_kept": kept_groups, "oversized_groups": len(oversized),
                          "source_columns": cols}
