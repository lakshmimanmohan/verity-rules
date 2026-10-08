"""Stateless entry point for rule learning. No file reads or writes."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from .column_classification import classify_columns
from .rule_learner import (
    build_profiles, kmeans_gap_threshold, learn_reliability,
    normalize_discriminative_weights, numeric_series_stats,
    record_id_from_row, weighted_pair_similarity,
)

CONTRACT_VERSION = "1.0"


def learn_rules(
    records: list[dict[str, Any]],
    embeddings: dict[str, list[float]] | None = None,
    pair_sample: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)
    if len(df) < 2:
        raise ValueError("Need at least 2 records.")

    cls = classify_columns(df)
    rows = df.to_dict(orient="records")
    content = [h for h in df.columns if h in set(cls["content_columns"])]
    if not content:
        raise ValueError("No content columns after classification.")
    profiles = build_profiles(content, rows, set(cls["metadata_columns"]))
    normalize_discriminative_weights(profiles)
    lo_hi = {p.name: numeric_series_stats(p.values) for p in profiles}

    rid_col = str(cls.get("record_id_column") or "").strip()
    if not rid_col:
        raise ValueError("Could not identify a record id column.")
    row_ids = [record_id_from_row(r, rid_col) for r in rows]

    n = len(rows)
    total = n * (n - 1) // 2
    if total <= pair_sample:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        s: set[tuple[int, int]] = set()
        while len(s) < pair_sample:
            i, j = random.randrange(n), random.randrange(n)
            if i != j:
                s.add((min(i, j), max(i, j)))
        pairs = list(s)

    sims = np.array([
        weighted_pair_similarity(rows[i], rows[j], profiles, lo_hi,
                                 embeddings=embeddings, row_ids=(row_ids[i], row_ids[j]))
        for i, j in pairs
    ], dtype=np.float64)

    rel = learn_reliability(pairs, rows, profiles, lo_hi)
    for p in profiles:
        p.reliability_score = float(rel.get(p.name, 1.0))
    comps = [max(1e-9, p.normalized_weight * p.reliability_score) for p in profiles]
    total_c = sum(comps)
    for p, c in zip(profiles, comps):
        p.composite_weight = c / total_c

    try:
        threshold, meta = kmeans_gap_threshold(sims)
        method = "kmeans_midpoint"
    except Exception as exc:  # noqa: BLE001
        threshold, meta, method = float(np.median(sims)), {"error": str(exc)}, "median_fallback"

    return {
        "contract_version": CONTRACT_VERSION,
        "n_records": n,
        "n_pairs_sampled": len(pairs),
        "record_id_column": rid_col,
        "embeddings_used": bool(embeddings),
        "columns_excluded": sorted(cls["metadata_columns"]),
        "threshold": {"value": round(float(threshold), 6), "method": method,
                      "details": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                  for k, v in meta.items()}},
        "similarity_stats": {"min": float(sims.min()), "max": float(sims.max()),
                             "mean": float(sims.mean()), "std": float(sims.std())},
        "columns": {
            p.name: {
                "semantic_type": p.semantic_type,
                "uniqueness_ratio": round(p.uniqueness_ratio, 6),
                "null_rate": round(p.null_rate, 6),
                "reliability_score": round(p.reliability_score, 6),
                "composite_weight": round(p.composite_weight, 6),
            } for p in profiles
        },
    }
