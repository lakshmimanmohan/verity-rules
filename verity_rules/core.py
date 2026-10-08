"""Stateless entry point for rule learning. No file reads or writes."""
from __future__ import annotations

import random
import re
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .column_classification import classify_columns
from .rule_learner import (
    build_profiles, comparison_method, kmeans_gap_threshold, learn_reliability,
    learn_reliability_v2,
    normalize_discriminative_weights, numeric_series_stats,
    is_null, record_id_from_row, weighted_pair_similarity,
)

CONTRACT_VERSION = "1.0"

CANDIDATE_MIN_UNIQUENESS = 0.5
CANDIDATE_GROUP_MIN, CANDIDATE_GROUP_MAX = 2, 10
CANDIDATE_CAP = 20_000
N_LIKELY_MATCH, N_UNCLEAR, N_LIKELY_NON_MATCH = 30, 40, 30
MAX_PER_GROUP = 2
NO_CANDIDATES_NOTE = ("No candidate pairs found (no column with enough distinct values shares a value "
                      "between 2 and 10 records); likely_match and unclear are empty, separability "
                      "could not be computed and reliability is not learned.")
RELIABILITY_HIGH, RELIABILITY_MID = 0.85, 0.65  # reason wording bands


def _build_candidates(rows, profiles) -> dict[tuple[int, int], tuple[str, str]]:
    """Pairs of rows that share a value in a high-uniqueness content column.
    Maps (i, j) -> group key (column, value); the first group found wins."""
    cands: dict[tuple[int, int], tuple[str, str]] = {}
    for p in profiles:
        if p.uniqueness_ratio < CANDIDATE_MIN_UNIQUENESS:
            continue
        groups: dict[str, list[int]] = {}
        for idx, v in enumerate(p.values):
            if is_null(v):
                continue
            groups.setdefault(str(v).strip().lower(), []).append(idx)
        for key in sorted(groups):
            members = groups[key]
            if not CANDIDATE_GROUP_MIN <= len(members) <= CANDIDATE_GROUP_MAX:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    cands.setdefault((members[a], members[b]), (p.name, key))
    return cands


def _clean_value(v: Any) -> Any:
    """Plain Python value for the public result: NaN, Infinity, None and pandas NA become None."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _round_floats(obj: Any) -> Any:
    """Round every float in a nested structure to six decimals; non-finite values become None."""
    if isinstance(obj, (float, np.floating)):
        return _clean_value(round(float(obj), 6))
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def _column_reason(p) -> str:
    """One sentence describing a kept column from uniqueness, nulls and reliability."""
    uniq = ("highly distinctive values" if p.uniqueness_ratio >= 0.7
            else "moderately distinctive values" if p.uniqueness_ratio >= 0.3
            else "many repeated values")
    nulls = ("rarely empty" if p.null_rate < 0.1
             else "sometimes empty" if p.null_rate < 0.4
             else "often empty")
    rel = ("agrees well across likely duplicates" if p.reliability_score >= RELIABILITY_HIGH
           else "usually agrees across likely duplicates" if p.reliability_score >= RELIABILITY_MID
           else "often disagrees across likely duplicates")
    return (f"{uniq.capitalize()} ({p.uniqueness_ratio:.0%} unique), {nulls} "
            f"({p.null_rate:.0%} null), and {rel} (reliability {p.reliability_score:.2f}).")


USER_EXCLUDED_RULE = "user_excluded"

_EXCLUDED_SENTENCES = {
    "reserved_name": "Its name marks it as a label, target or internal column.",
    "primary_key": "Almost every value is different, so it identifies a row rather than an entity.",
    "answer_key": "Most values look like generated labels (WORD_number or UNKNOWN_...).",
    "boolean_flag": "It has two or three values and its name suggests a yes/no flag.",
    "degenerate_field": "Every record has the same value, so it cannot tell records apart.",
    "near_constant_field": "It has very few distinct values, so it cannot tell records apart.",
    "record_id_column": "It is the record id column.",
    USER_EXCLUDED_RULE: "Excluded at your request.",
}


def _rule_tags(raw: str) -> list[str]:
    """The classifier's rule tags in a raw reasoning string ('[tag] detail | [tag] detail')."""
    return re.findall(r"\[(\w+)\]", raw)


def _excluded_sentence(raw: str) -> str:
    """Turn the classifier's raw reason into plain sentences."""
    out: list[str] = []
    for part in raw.removeprefix("excluded: ").split(" | "):
        m = re.match(r"\[(\w+)\]\s*(.*)", part.strip())
        if not m:
            out.append(part.strip().rstrip(".") + ".")
            continue
        tag, detail = m.groups()
        if tag == "system_process_field":
            words = re.search(r"name contains: ([^;)]*)", detail)
            if "100% unique" in detail:
                sentence = (f"Its name contains {words.group(1).strip() if words else 'id/key/code'}"
                            " and every value is different.")
            else:
                sentence = ("Its name suggests a system or process field"
                            + (f" (it contains: {words.group(1).strip()})" if words else "") + ".")
            out.append(sentence)
        else:
            out.append(_EXCLUDED_SENTENCES.get(tag, detail.rstrip(".") + "."))
    return " ".join(out)


def _excluded_entry(name: str, raw: str) -> dict[str, Any]:
    """Public entry for an excluded column: plain sentence plus the raw rule tags."""
    return {"name": name, "reason": _excluded_sentence(raw), "rule": _rule_tags(raw)}


def _apply_column_overrides(cls: dict[str, Any], columns: list[str],
                            exclude: list[str], keep: list[str], notes: list[str]) -> None:
    """Apply the user's exclude/keep lists to the classification, in place.
    Raises ValueError (naming exclude_columns / keep_columns) for unknown names, a name in
    both lists, or keeping the record id column."""
    both = sorted(set(exclude) & set(keep))
    if both:
        raise ValueError("exclude_columns and keep_columns both list: " + ", ".join(both))
    for option, names in (("exclude_columns", exclude), ("keep_columns", keep)):
        unknown = sorted(set(names) - set(columns))
        if unknown:
            raise ValueError(f"{option}: no such column: " + ", ".join(unknown))
    rid = cls["record_id_column"]
    if rid in keep:
        raise ValueError(f"keep_columns: {rid} is the record id column and cannot be kept")
    metadata, content = set(cls["metadata_columns"]), set(cls["content_columns"])
    for c in sorted(set(keep) & metadata):
        notes.append(f"{c}: Verity would have excluded it ({_excluded_sentence(cls['reasoning'][c])}) "
                     "but it was kept at your request.")
        metadata.discard(c)
        content.add(c)
    for c in sorted(set(exclude) & content):
        metadata.add(c)
        content.discard(c)
        cls["reasoning"][c] = f"excluded: [{USER_EXCLUDED_RULE}] excluded by the user"
    cls["metadata_columns"] = sorted(metadata)
    cls["content_columns"] = sorted(content)


def _pair_view(i, j, rows, row_ids, content) -> dict[str, Any]:
    def vals(k):
        return {c: _clean_value(rows[k].get(c)) for c in content}
    return {"record_id_a": row_ids[i], "record_id_b": row_ids[j],
            "values_a": vals(i), "values_b": vals(j)}


def learn_rules(
    records: list[dict[str, Any]],
    embeddings: dict[str, list[float]] | None = None,
    pair_sample: int = 1000,
    seed: int = 42,
    debug: bool = False,
    reliability: str = "v2",
    exclude_columns: list[str] | None = None,
    keep_columns: list[str] | None = None,
) -> dict[str, Any]:
    if reliability not in ("v1", "v2"):
        raise ValueError(f"reliability must be 'v1' or 'v2', got {reliability!r}")
    random.seed(seed)
    np.random.seed(seed)

    df = pd.DataFrame(records)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)
    if len(df) < 2:
        raise ValueError("Need at least 2 records.")

    cls = classify_columns(df)
    notes: list[str] = []
    _apply_column_overrides(cls, list(df.columns), list(exclude_columns or []),
                            list(keep_columns or []), notes)
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

    # Candidate pairs: random pairs are almost all non-matches.
    cand_map = _build_candidates(rows, profiles)
    cand_pairs = sorted(cand_map)
    if len(cand_pairs) > CANDIDATE_CAP:
        cand_pairs = sorted(random.sample(cand_pairs, CANDIDATE_CAP))
        notes.append(f"Candidate pairs capped at {CANDIDATE_CAP} by sampling.")
    rel_info: dict[str, Any] = {}
    if reliability == "v2":
        rel, rel_info = learn_reliability_v2(
            cand_pairs, rows, profiles, lo_hi, embeddings=embeddings, row_ids=row_ids)
        notes.extend(rel_info["notes"])
    else:
        rel = learn_reliability(pairs, rows, profiles, lo_hi)
    for p in profiles:
        p.reliability_score = float(rel.get(p.name, 1.0))
    comps = [max(1e-9, p.normalized_weight * p.reliability_score) for p in profiles]
    total_c = sum(comps)
    for p, c in zip(profiles, comps):
        p.composite_weight = c / total_c

    # Score with the published composite weights (reliability does not use these scores).
    sims = np.array([
        weighted_pair_similarity(rows[i], rows[j], profiles, lo_hi,
                                 embeddings=embeddings, row_ids=(row_ids[i], row_ids[j]),
                                 weight_attr="composite_weight")
        for i, j in pairs
    ], dtype=np.float64)

    try:
        threshold, meta = kmeans_gap_threshold(sims)
        method = "kmeans_midpoint"
    except Exception as exc:  # noqa: BLE001
        threshold, meta, method = float(np.median(sims)), {"error": str(exc)}, "median_fallback"

    cand_sims = np.array([
        weighted_pair_similarity(rows[i], rows[j], profiles, lo_hi,
                                 embeddings=embeddings, row_ids=(row_ids[i], row_ids[j]),
                                 weight_attr="composite_weight")
        for i, j in cand_pairs
    ], dtype=np.float64)

    rand_mean = float(sims.mean())
    separability: dict[str, Any] = {"label": None, "candidate_top_decile_mean": None,
                                    "random_pair_mean": None, "gap": None}
    calibration: dict[str, list] = {"likely_match": [], "unclear": [], "likely_non_match": []}
    if len(cand_pairs) == 0:
        notes.append(NO_CANDIDATES_NOTE)
    else:
        floor = float(np.percentile(cand_sims, 90))
        top = cand_sims >= floor
        high_mean = float(cand_sims[top].mean())
        gap = high_mean - rand_mean
        label = "clear" if gap >= 0.30 else "moderate" if gap >= 0.15 else "weak"
        separability = {"label": label, "candidate_top_decile_mean": high_mean,
                        "random_pair_mean": rand_mean, "gap": gap}

        order = list(range(len(cand_pairs)))
        random.shuffle(order)
        per_group: dict[tuple[str, str], int] = {}
        for k in order:
            if not top[k] or len(calibration["likely_match"]) >= N_LIKELY_MATCH:
                continue
            g = cand_map[cand_pairs[k]]
            if per_group.get(g, 0) >= MAX_PER_GROUP:
                continue
            per_group[g] = per_group.get(g, 0) + 1
            calibration["likely_match"].append(cand_pairs[k])
        mid = [k for k in order if rand_mean <= cand_sims[k] < floor]
        calibration["unclear"] = [cand_pairs[k] for k in mid[:N_UNCLEAR]]

    low = [pairs[k] for k in range(len(pairs)) if sims[k] < rand_mean]
    random.shuffle(low)
    calibration["likely_non_match"] = low[:N_LIKELY_NON_MATCH]

    want = {"likely_match": N_LIKELY_MATCH, "unclear": N_UNCLEAR,
            "likely_non_match": N_LIKELY_NON_MATCH}
    for cat, pl in calibration.items():
        if len(pl) < want[cat]:
            notes.append(f"{cat}: only {len(pl)} of {want[cat]} requested pairs available.")
        calibration[cat] = [_pair_view(i, j, rows, row_ids, content) for i, j in pl]

    result = {
        "contract_version": CONTRACT_VERSION,
        "n_records": n,
        "n_pairs_sampled": len(pairs),
        "record_id_column": rid_col,
        "embeddings_used": bool(embeddings),
        "run": {"verity_version": __version__, "seed": seed,
                "n_candidate_pairs": len(cand_pairs)},
        "columns_excluded": [
            _excluded_entry(c, cls["reasoning"].get(c, "excluded as metadata"))
            for c in sorted(cls["metadata_columns"])
        ],
        "separability": separability,
        "calibration_pairs": calibration,
        "notes": notes,
        "columns": {
            p.name: {
                "semantic_type": p.semantic_type,
                "comparison_method": comparison_method(p, bool(embeddings)),
                "uniqueness_ratio": round(p.uniqueness_ratio, 6),
                "null_rate": round(p.null_rate, 6),
                "reliability_score": round(p.reliability_score, 6),
                "composite_weight": round(p.composite_weight, 6),
                "reason": _column_reason(p),
            } for p in profiles
        },
    }
    result = _round_floats(result)
    if debug:
        if reliability == "v2":
            result["reliability_debug"] = {
                "method": "v2", "anchor_count": rel_info["anchor_count"],
                "min_agree_used": rel_info["min_agree_used"]}
        result["candidate_pairs"] = len(cand_pairs)
        result["threshold"] = {
            "value": round(float(threshold), 6), "method": method,
            "details": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in meta.items()}}
        result["similarity_stats"] = {
            "min": float(sims.min()), "max": float(sims.max()),
            "mean": rand_mean, "std": float(sims.std())}
    return result
