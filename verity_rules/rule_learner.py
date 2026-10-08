"""
Discover schema, discriminative weights, similarity threshold, and field reliability
from an arbitrary CSV (no assumed column names).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .column_classification import (
    classify_columns,
    global_column_classification_path,
    print_classification_report,
    write_file_scoped_column_classification,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = None
OUT_WEIGHTS = PROJECT_ROOT / "data" / "attribute_weights.json"
OUT_THRESHOLD = PROJECT_ROOT / "data" / "learned_threshold.json"
OUT_PLOT = PROJECT_ROOT / "data" / "threshold_similarity_distribution.png"
OUT_RULES = PROJECT_ROOT / "data" / "learned_rules.md"
GLOBAL_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings_global.pkl"
EMBEDDINGS_STORE_META_KEY = "__embeddings_store_meta__"

PAIR_SAMPLE = 1000
RANDOM_SEED = 42
HYBRID_EMBED_WEIGHT = 0.65
HYBRID_CHAR_WEIGHT = 0.35

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z\s\-'.]*$")
EMAIL_RE = re.compile(r"^[^@\s]+@.+\..+$")

@dataclass
class ColumnProfile:
    name: str
    semantic_type: str
    physical_kind: str
    values: list[Any] = field(repr=False)
    uniqueness_ratio: float = 0.0
    null_rate: float = 0.0
    discriminative_score: float = 0.0
    normalized_weight: float = 0.0
    reliability_score: float = 1.0
    composite_weight: float = 0.0


def is_null(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "n/a"}


def clean_str(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def token_key(v: Any) -> str:
    if is_null(v):
        return "__NULL__"
    return clean_str(v).lower()


def resolve_input_csv(input_path: str | Path) -> Path:
    """Resolve CSV path; relative paths are taken from ``PROJECT_ROOT``."""
    p = Path(input_path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row.")
        headers = [h for h in reader.fieldnames if h is not None]
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append({h: row.get(h, "") for h in headers})
    if not headers or not rows:
        raise SystemExit("CSV is empty or has no data rows.")
    return headers, rows


def try_parse_number(s: str) -> float | None:
    t = s.replace(",", "").strip()
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def try_parse_date(s: str) -> datetime | None:
    t = s.strip()
    if not t:
        return None
    fmts = (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d-%b-%Y",
        "%b %d, %Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(t.replace("Z", "+00:00")[:19])
    except ValueError:
        return None


def proportion_parseable(values: list[Any], kind: str) -> float:
    nn = [clean_str(v) for v in values if not is_null(v)]
    if not nn:
        return 0.0
    ok = 0
    for s in nn:
        if kind == "num" and try_parse_number(s) is not None:
            ok += 1
        elif kind == "date" and try_parse_date(s) is not None:
            ok += 1
    return ok / len(nn)


def proportion_email_like(values: list[Any]) -> float:
    nn = [clean_str(v) for v in values if not is_null(v)]
    if not nn:
        return 0.0
    ok = sum(1 for s in nn if "@" in s and EMAIL_RE.match(s.replace("(at)", "@")) is not None)
    return ok / len(nn)


def proportion_name_like(values: list[Any]) -> float:
    nn = [clean_str(v) for v in values if not is_null(v)]
    if not nn:
        return 0.0
    ok = 0
    for s in nn:
        if any(ch.isdigit() for ch in s):
            continue
        if s.isupper() or s.islower():
            # allow but weight lower — still can be names
            if NAME_RE.match(s):
                ok += 1
            continue
        if NAME_RE.match(s) and re.search(r"[a-z]", s) and re.search(r"[A-Z]", s):
            ok += 1
        elif NAME_RE.match(s):
            ok += 1
    return ok / len(nn)


# Matches values that start with 1-5 digits (a street number) followed by at least
# one alphabetic word — e.g. "123 Main St", "4 Rue de Rivoli", "10-12 High Street".
# The leading digit block may be hyphenated (e.g. "10-12") and may have an optional
# suffix letter ("123A"). Does NOT match pure numeric strings or PO Box patterns.
_STREET_ADDRESS_RE = re.compile(
    r"^\d{1,5}[A-Za-z]?(?:-\d{1,5})?[\s,]+[A-Za-z]",
    re.IGNORECASE,
)


def proportion_street_address_like(values: list[Any]) -> float:
    """
    Fraction of non-null values that look like a street address:
    starts with 1-5 digits (optionally hyphenated or letter-suffixed),
    followed by whitespace/comma and at least one alphabetic character.
    Examples that match: '123 Main St', '4B Elm Ave', '10-12 High Rd', '1, Park Lane'.
    Examples that do not: '10001', 'P.O. Box 5', 'Suite 400', 'ACME Corp'.
    """
    nn = [clean_str(v) for v in values if not is_null(v)]
    if not nn:
        return 0.0
    ok = sum(1 for s in nn if _STREET_ADDRESS_RE.match(s))
    return ok / len(nn)


def column_uniqueness_and_nulls(values: list[Any], n_rows: int) -> tuple[float, float]:
    keys = [token_key(v) for v in values]
    nulls = sum(1 for v in values if is_null(v))
    null_rate = nulls / n_rows if n_rows else 0.0
    uniq = len(set(keys))
    uniqueness_ratio = uniq / n_rows if n_rows else 0.0
    return uniqueness_ratio, null_rate


def infer_physical_kind(colname: str, values: list[Any]) -> str:
    if proportion_parseable(values, "date") >= 0.85:
        return "DATE"
    if proportion_parseable(values, "num") >= 0.90:
        return "NUMBER"
    if proportion_email_like(values) >= 0.65:
        return "EMAIL"
    return "STRING"


def infer_semantic_type(
    colname: str,
    values: list[Any],
    uniqueness_ratio: float,
    physical_kind: str,
    is_metadata: bool = False,
) -> str:
    date_p = proportion_parseable(values, "date")
    if date_p >= 0.85:
        return "DATE"
    # Value-pattern check: low distinct value count means categorical/lifecycle field.
    # Fewer than 10 distinct non-null values is categorical by definition regardless
    # of uniqueness ratio or column name.
    nn_values = [clean_str(v) for v in values if not is_null(v)]
    n_distinct = len(set(nn_values))
    # Low cardinality is categorical regardless of physical kind (covers numeric codes too).
    descriptive_cat = n_distinct < 10
    # Value-pattern check: fields where most values look like street addresses
    # get their own semantic type, separate from generic DESCRIPTIVE. This is
    # detected purely from value shape (leading digits + alpha text), so it
    # applies identically to a location's address line, a vendor's mailing
    # address, or any other free-text address-shaped field across domains.
    # Splitting STREET_ADDRESS out of DESCRIPTIVE lets downstream matching
    # logic apply address-specific normalization (abbreviation equivalence,
    # PO Box handling) instead of generic categorical-field treatment.
    descriptive_address = (
        physical_kind == "STRING"
        and proportion_street_address_like(values) >= 0.60
    )
    if descriptive_address:
        return "STREET_ADDRESS"
    if descriptive_cat or (
        uniqueness_ratio < 0.08 and physical_kind != "EMAIL"
    ):
        return "DESCRIPTIVE"

    if physical_kind == "EMAIL" or proportion_email_like(values) >= 0.70:
        if uniqueness_ratio > 0.90:
            return "IDENTIFIER"
        if uniqueness_ratio >= 0.60:
            return "QUASI_IDENTIFIER"
        return "DESCRIPTIVE"
    # Value-pattern check: name-like fields are detected purely from value patterns.
    # proportion_name_like already handles this; the threshold is lowered slightly
    # (0.50) to catch mixed datasets where some records have abbreviated names.
    nn_vals = [clean_str(v) for v in values if not is_null(v)]
    avg_words = sum(len(s.split()) for s in nn_vals) / len(nn_vals) if nn_vals else 1.0

    if physical_kind != "NUMBER" and (
        proportion_name_like(values) >= 0.50
        and uniqueness_ratio >= 0.18
        and uniqueness_ratio < 0.80
        and physical_kind == "STRING"
        and avg_words >= 1.5
    ):
        return "NAME"

    if uniqueness_ratio > 0.90 or (
        physical_kind == "NUMBER" and uniqueness_ratio > 0.85
    ):
        # Guard: high-cardinality free text is DESCRIPTIVE not IDENTIFIER.
        # Multi-word values with spaces are never keys — data-driven on value
        # patterns only, no column name assumptions.
        if physical_kind == "STRING":
            nn_count = sum(1 for v in values if not is_null(v))
            if nn_count > 0:
                avg_word_count = (
                    sum(len(clean_str(v).split()) for v in values if not is_null(v))
                    / nn_count
                )
                space_ratio = (
                    sum(
                        1
                        for v in values
                        if not is_null(v) and " " in clean_str(v)
                    )
                    / nn_count
                )
                if avg_word_count >= 3 and space_ratio >= 0.70:
                    return "DESCRIPTIVE"
        return "IDENTIFIER"

    if uniqueness_ratio >= 0.60:
        return "QUASI_IDENTIFIER"
    return "DESCRIPTIVE"


def build_profiles(
    headers: list[str],
    rows: list[dict[str, Any]],
    metadata_column_names: set[str] | None = None,
) -> list[ColumnProfile]:
    n = len(rows)
    profiles: list[ColumnProfile] = []
    meta = metadata_column_names or set()
    for h in headers:
        colvals = [r.get(h, "") for r in rows]
        uq, nr = column_uniqueness_and_nulls(colvals, n)
        pk = infer_physical_kind(h, colvals)
        sem = infer_semantic_type(h, colvals, uq, pk, is_metadata=h in meta)
        if str(sem).strip().upper() == "IDENTIFIER":
            pk = "STRING"
        disc = uq * (1.0 - nr)
        profiles.append(
            ColumnProfile(
                name=h,
                semantic_type=sem,
                physical_kind=pk,
                values=colvals,
                uniqueness_ratio=uq,
                null_rate=nr,
                discriminative_score=disc,
            )
        )
    return profiles


def normalize_discriminative_weights(profiles: list[ColumnProfile]) -> None:
    raw = [max(0.0, p.discriminative_score) for p in profiles]
    s = sum(raw)
    if s <= 0:
        u = 1.0 / max(1, len(profiles))
        for p in profiles:
            p.normalized_weight = u
        return
    for p in profiles:
        p.normalized_weight = p.discriminative_score / s


def char_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower()), set(b.lower())
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / uni if uni else 1.0


def load_global_embeddings(path: Path) -> dict[str, list[float]]:
    """Load record embeddings from ``embeddings_global.pkl`` (skip reserved meta key)."""
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[float]] = {}
    for k, v in raw.items():
        sk = str(k).strip()
        if not sk or sk == EMBEDDINGS_STORE_META_KEY:
            continue
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            out[sk] = [float(x) for x in v]
        elif hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
            try:
                out[sk] = [float(x) for x in v]  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0 or math.isnan(na) or math.isnan(nb):
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    if math.isnan(c):
        return 0.0
    return max(-1.0, min(1.0, c))


def _embedding_for_record(
    embeddings: dict[str, list[float]], record_id: str | None
) -> np.ndarray | None:
    if not embeddings or not record_id:
        return None
    vec = embeddings.get(str(record_id).strip())
    if vec is None:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def hybrid_field_similarity(
    va: Any,
    vb: Any,
    record_emb_i: np.ndarray | None,
    record_emb_j: np.ndarray | None,
) -> float:
    cj = char_jaccard(clean_str(va), clean_str(vb))
    if record_emb_i is None or record_emb_j is None:
        return cj
    cos = cosine_similarity(record_emb_i, record_emb_j)
    return HYBRID_EMBED_WEIGHT * cos + HYBRID_CHAR_WEIGHT * cj


def record_id_from_row(row: dict[str, Any], rid_col: str) -> str:
    key = str(rid_col).strip()
    if key in row:
        val = row[key]
    else:
        lk = key.lower()
        val = next(
            (row[k] for k in row if str(k).strip().lower() == lk),
            None,
        )
    if is_null(val):
        return ""
    return str(val).strip()


def numeric_series_stats(values: list[Any]) -> tuple[float | None, float | None]:
    nums: list[float] = []
    for v in values:
        if is_null(v):
            continue
        x = try_parse_number(clean_str(v))
        if x is None:
            return None, None
        nums.append(x)
    if not nums:
        return None, None
    return min(nums), max(nums)


def numeric_similarity(va: Any, vb: Any, lo: float, hi: float) -> float:
    xa = try_parse_number(clean_str(va))
    xb = try_parse_number(clean_str(vb))
    if xa is None or xb is None:
        return 0.0
    if hi == lo:
        return 1.0 if math.isclose(xa, xb, rel_tol=1e-9, abs_tol=1e-9) else 0.0
    sa = (xa - lo) / (hi - lo)
    sb = (xb - lo) / (hi - lo)
    return float(max(0.0, 1.0 - abs(sa - sb)))


def date_similarity(va: Any, vb: Any) -> float:
    da = try_parse_date(clean_str(va))
    db = try_parse_date(clean_str(vb))
    if da is None or db is None:
        return 0.0
    diff_days = abs((da - db).days)
    return float(max(0.0, 1.0 - min(1.0, diff_days / 3650.0)))  # 10-year scale


def email_similarity(va: Any, vb: Any) -> float:
    a = clean_str(va).lower().replace("(at)", "@")
    b = clean_str(vb).lower().replace("(at)", "@")
    if "@" not in a or "@" not in b:
        return char_jaccard(a, b)
    la, da = a.split("@", 1)
    lb, db = b.split("@", 1)
    return 0.5 * char_jaccard(la, lb) + 0.5 * char_jaccard(da, db)


def field_similarity(
    va: Any,
    vb: Any,
    prof: ColumnProfile,
    lo_hi: tuple[float | None, float | None],
    *,
    record_emb_i: np.ndarray | None = None,
    record_emb_j: np.ndarray | None = None,
) -> float:
    if is_null(va) or is_null(vb):
        return 0.0
    semantic_type = str(prof.semantic_type).strip().upper()
    physical_type = str(prof.physical_kind).strip().upper()
    if semantic_type == "IDENTIFIER":
        return 1.0 if clean_str(va) == clean_str(vb) else 0.0
    if physical_type == "EMAIL":
        return email_similarity(va, vb)
    if physical_type == "DATE":
        return date_similarity(va, vb)
    if physical_type == "NUMBER":
        lo, hi = lo_hi
        if lo is None or hi is None:
            return char_jaccard(clean_str(va), clean_str(vb))
        return numeric_similarity(va, vb, lo, hi)
    if semantic_type in ("DESCRIPTIVE", "NAME", "STREET_ADDRESS") and physical_type == "STRING":
        return hybrid_field_similarity(va, vb, record_emb_i, record_emb_j)
    return char_jaccard(clean_str(va), clean_str(vb))


def weighted_pair_similarity(
    row_i: dict[str, Any],
    row_j: dict[str, Any],
    profiles: list[ColumnProfile],
    lo_hi_map: dict[str, tuple[float | None, float | None]],
    embeddings: dict[str, list[float]] | None = None,
    row_ids: tuple[str, str] | None = None,
) -> float:
    record_emb_i: np.ndarray | None = None
    record_emb_j: np.ndarray | None = None
    if embeddings and row_ids is not None:
        record_emb_i = _embedding_for_record(embeddings, row_ids[0])
        record_emb_j = _embedding_for_record(embeddings, row_ids[1])

    num = 0.0
    den = 0.0
    for p in profiles:
        vi, vj = row_i.get(p.name), row_j.get(p.name)
        if is_null(vi) or is_null(vj):
            continue
        w = p.normalized_weight
        s = field_similarity(
            vi,
            vj,
            p,
            lo_hi_map[p.name],
            record_emb_i=record_emb_i,
            record_emb_j=record_emb_j,
        )
        num += w * s
        den += w
    return num / den if den > 0 else 0.0


def exact_match(va: Any, vb: Any, prof: ColumnProfile, lo_hi: tuple[float | None, float | None]) -> bool:
    if is_null(va) or is_null(vb):
        return False
    if prof.physical_kind == "NUMBER":
        xa, xb = try_parse_number(clean_str(va)), try_parse_number(clean_str(vb))
        if xa is None or xb is None:
            return token_key(va) == token_key(vb)
        return math.isclose(xa, xb, rel_tol=1e-9, abs_tol=1e-6)
    return token_key(va) == token_key(vb)


def exact_duplicate_pairs(
    rows_by_id: dict[str, dict[str, str]],
    content_columns_lower: frozenset[str],
    semantic_by_col_lower: dict[str, str],
) -> list[tuple[str, str]]:
    """
    Group records by a normalized hash of their content fields.
    Fields classified as QUASI_IDENTIFIER are excluded from the hash key —
    they have high uniqueness and formatting variance (postcodes, phone numbers,
    partial codes) that makes identical records appear different.
    Only NAME and DESCRIPTIVE fields are used: they are stable across
    duplicate copies of the same entity.
    No column names are inspected.
    """
    def norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]", "", s)
        return re.sub(r" +", " ", s)

    # Use NAME + DESCRIPTIVE + STREET_ADDRESS fields for the hash.
    # QUASI_IDENTIFIER columns (postcodes, phone numbers, codes) vary across
    # duplicates and break the hash. IDENTIFIER columns are unique per record
    # by definition and must never appear in a duplicate-detection key.
    # STREET_ADDRESS is included explicitly (not folded into DESCRIPTIVE)
    # because address matching needs separate normalization logic downstream
    # (abbreviation/PO Box handling) — see normalize_street_address().
    hash_fields = sorted(
        f for f in content_columns_lower
        if str(semantic_by_col_lower.get(f, "DESCRIPTIVE")).strip().upper()
        in {"NAME", "DESCRIPTIVE", "STREET_ADDRESS"}
    )
    if not hash_fields:
        return []

    buckets: dict[str, list[str]] = defaultdict(list)
    for rid, row in rows_by_id.items():
        parts = [norm(str(row.get(f, "") or "")) for f in hash_fields]
        non_empty = sum(1 for p in parts if p)
        if non_empty < 2:
            continue
        key = hashlib.md5("|".join(parts).encode()).hexdigest()
        buckets[key].append(rid)

    pairs: list[tuple[str, str]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, a in enumerate(sorted(members)):
            for b in sorted(members)[i + 1:]:
                pairs.append((a, b))
    return pairs


def learn_reliability(
    pairs: list[tuple[int, int]],
    rows: list[dict[str, Any]],
    profiles: list[ColumnProfile],
    lo_hi_map: dict[str, tuple[float | None, float | None]],
) -> dict[str, float]:
    support: dict[str, float] = defaultdict(float)
    conflict: dict[str, float] = defaultdict(float)
    w_by_col = {p.name: max(p.normalized_weight, 1e-6) for p in profiles}

    for i, j in pairs:
        ri, rj = rows[i], rows[j]
        matched_fields: list[ColumnProfile] = []
        for p in profiles:
            vi, vj = ri.get(p.name), rj.get(p.name)
            if exact_match(vi, vj, p, lo_hi_map[p.name]):
                matched_fields.append(p)
        if not matched_fields:
            continue
        anchor_w_sum = sum(w_by_col[m.name] for m in matched_fields)
        if anchor_w_sum <= 0:
            continue
        for p in profiles:
            vn = p.name
            vi, vj = ri.get(vn), rj.get(vn)
            if is_null(vi) or is_null(vj):
                continue
            if exact_match(vi, vj, p, lo_hi_map[p.name]):
                support[vn] += anchor_w_sum
            else:
                conflict[vn] += anchor_w_sum

    reliability: dict[str, float] = {}
    for p in profiles:
        vn = p.name
        s = support.get(vn, 0.0)
        c = conflict.get(vn, 0.0)
        reliability[vn] = float(s / (s + c + 1e-9))

    # squash to usable range — avoid degenerate zeros
    vals = list(reliability.values())
    lo, hi = min(vals), max(vals)
    if hi > lo:
        for k in reliability:
            reliability[k] = 0.1 + 0.9 * (reliability[k] - lo) / (hi - lo)
    else:
        for k in reliability:
            reliability[k] = 1.0
    return reliability


def kmeans_gap_threshold(sim_array: np.ndarray) -> tuple[float, dict[str, float]]:
    meta: dict[str, float] = {}
    X = sim_array.reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_SEED).fit(X)
    centers = sorted(float(c) for c in km.cluster_centers_.flatten())
    c0, c1 = centers[0], centers[1]
    threshold = (c0 + c1) / 2.0
    meta["cluster_center_low"] = c0
    meta["cluster_center_high"] = c1
    meta["inertia"] = float(km.inertia_)
    return threshold, meta


def print_schema_report(profiles: list[ColumnProfile]) -> None:
    print("\n" + "=" * 72)
    print("  DISCOVERED SCHEMA")
    print("=" * 72)
    for p in profiles:
        print(
            f"  {p.name:24}  semantic={p.semantic_type:16}  "
            f"physical={p.physical_kind:8}  "
            f"uniq={p.uniqueness_ratio:.3f}  null={p.null_rate:.3f}"
        )
    print("=" * 72 + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _fmt_opt_float(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int, float)):
        xf = float(x)
        if math.isnan(xf):
            return "n/a"
        return f"{xf:.4f}"
    return str(x)


def write_markdown(
    csv_path: Path,
    profiles: list[ColumnProfile],
    threshold: float,
    th_meta: dict[str, Any],
    pair_count: int,
    excluded_cols: list[str],
) -> None:
    lines = [
        "# Learned matching rules\n",
        f"Source file: `{csv_path}`  \n",
        f"Rows analyzed: `{th_meta.get('n_rows', '')}`\n\n",
    ]
    if excluded_cols:
        joined = ", ".join(f"`{c}`" for c in sorted(excluded_cols))
        lines.append(
            f"> **Excluded from learning** (hold-out / label-like columns): {joined}.  \n\n"
        )
    lines += [
        "## Schema discovery\n",
        "Columns were typed from value patterns (dates, emails, numeric parse rates, name-like strings) ",
        "and cardinality tiers (identifiers vs quasi-identifiers vs descriptive fields). ",
        "The weight table lists **columns used for learning** (evaluation / label-like fields removed).\n\n",
        "| Column | Semantic type | Physical kind | Uniq. ratio | Null % | Weight (disc.) |\n",
        "|--------|----------------|---------------|------------|-------|----------------|\n",
    ]
    ranked = sorted(profiles, key=lambda z: z.normalized_weight, reverse=True)
    for p in ranked:
        lines.append(
            f"| `{p.name}` | {p.semantic_type} | {p.physical_kind} "
            f"| {p.uniqueness_ratio:.3f} | {p.null_rate*100:.1f}% "
            f"| {p.normalized_weight:.3f} |\n"
        )

    lines += [
        "\n## Discriminative weights\n",
        "Each column receives `discriminative_score = uniqueness_ratio * (1 - null_rate)`. ",
        "Values are normalized to sum to **1.0** so they act as data-driven attribute weights ",
        "**without manual tuning**.\n\n",
        "### Strongest signals\n",
    ]
    for p in ranked[:5]:
        lines.append(
            f"- **`{p.name}`** normalized weight **{p.normalized_weight:.3f}**: "
            f"{p.uniqueness_ratio*100:.1f}% unique values, "
            f"{p.null_rate*100:.1f}% null — classified as **{p.semantic_type}**.\n"
        )

    lines += [
        "\n## Similarity aggregation\n",
        "- DESCRIPTIVE / NAME string columns: "
        f"{HYBRID_EMBED_WEIGHT:.0%} cosine (record embedding) + "
        f"{HYBRID_CHAR_WEIGHT:.0%} character-set Jaccard; "
        "falls back to Jaccard when embeddings are missing  \n",
        "- Other strings: character-set Jaccard similarity  \n",
        "- Numbers: min–max normalization per column; similarity `1 − |Δ_scaled|`  \n",
        "- Dates: ordinal distance scaled to ~10 years  \n",
        "- Email: separate Jaccard on local-part and domain, then averaged  \n\n",
        f"Random **{pair_count}** row pairs were scored with the normalized weights.\n\n",
        "## Learned match threshold\n",
        f"Pooled weighted similarity was clustered with **`KMeans(n_clusters=2)`** ",
        "(separating a low cluster from a high cluster).\n\n",
        f"- Lower cluster center: **`{_fmt_opt_float(th_meta.get('cluster_center_low'))}`**  \n",
        f"- Upper cluster center: **`{_fmt_opt_float(th_meta.get('cluster_center_high'))}`**  \n",
        f"- **`threshold = midpoint = {threshold:.4f}`** — pairs above this are treated as plausible matches.\n\n",
        "See **`data/threshold_similarity_distribution.png`** for the histogram with the cutoff.\n\n",
        "## Field reliability\n",
        "When pairs already agree on strongly weighted columns but disagree elsewhere, ",
        "the conflicting field accumulates penalties; fields that consistently co-agree ",
        "with anchors gain higher **`reliability_score`**.\n\n",
        "| Column | Reliability |\n",
        "|--------|------------|\n",
    ]
    for p in sorted(profiles, key=lambda z: z.reliability_score, reverse=True):
        lines.append(f"| `{p.name}` | {p.reliability_score:.3f} |\n")

    lines += [
        "\n## Composite suggestion\n",
        "`composite_weight` (normalized) blends discriminative uniqueness with reliability — ",
        "useful when some fields drift (addresses) while others stay stable (email/phone).\n",
    ]
    OUT_RULES.parent.mkdir(parents=True, exist_ok=True)
    OUT_RULES.write_text("".join(lines), encoding="utf-8")


def main(input_path: str | Path = DEFAULT_INPUT_PATH) -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    csv_path = resolve_input_csv(input_path)
    if not csv_path.is_file():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    print(f"Reading `{csv_path}`...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.where(pd.notna(df), None)
    classification = classify_columns(df)
    cls_out = csv_path.parent / "column_classification.json"
    print_classification_report(classification)

    headers = list(df.columns)
    rows = df.to_dict(orient="records")
    n_rows = len(rows)
    content_headers = [h for h in headers if h in set(classification["content_columns"])]
    if not content_headers:
        raise SystemExit("No content columns after classification — check your CSV.")
    profiles_full = build_profiles(
        content_headers, rows, set(classification["metadata_columns"])
    )
    classification["semantic_types"] = {
        str(p.name).strip(): str(p.semantic_type).strip().upper() for p in profiles_full
    }
    write_file_scoped_column_classification(cls_out, classification)
    print(f"Saved `{cls_out}`")
    print(f"Updated global column list: `{global_column_classification_path()}`")
    print_schema_report(profiles_full)

    excluded_cols = sorted(classification["metadata_columns"])
    profiles = list(profiles_full)
    if excluded_cols:
        print(
            "Excluded from weight/threshold/reliability learning (classified metadata): "
            + ", ".join(sorted(excluded_cols))
            + "\n"
        )
    if not profiles:
        raise SystemExit("Every column was excluded from learning — relax classification rules.")

    normalize_discriminative_weights(profiles)

    lo_hi_map: dict[str, tuple[float | None, float | None]] = {}
    for p in profiles:
        lo_hi_map[p.name] = numeric_series_stats(p.values)

    rid_col = str(classification.get("record_id_column") or "").strip()
    if not rid_col:
        raise SystemExit("column classification is missing record_id_column.")
    row_ids = [record_id_from_row(r, rid_col) for r in rows]
    embeddings = load_global_embeddings(GLOBAL_EMBEDDINGS_PATH)

    all_pairs_count = n_rows * (n_rows - 1) // 2
    n_pairs = min(PAIR_SAMPLE, all_pairs_count)
    if n_pairs <= 0:
        raise SystemExit("Not enough rows to sample pairs.")

    if all_pairs_count <= n_pairs:
        idx_pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
    else:
        sampled_set: set[tuple[int, int]] = set()
        while len(sampled_set) < n_pairs:
            i = random.randrange(n_rows)
            j = random.randrange(n_rows)
            if i != j:
                sampled_set.add((min(i, j), max(i, j)))
        idx_pairs = list(sampled_set)

    sims: list[float] = []
    for i, j in idx_pairs:
        sims.append(
            weighted_pair_similarity(
                rows[i],
                rows[j],
                profiles,
                lo_hi_map,
                embeddings=embeddings,
                row_ids=(row_ids[i], row_ids[j]),
            )
        )
    sim_arr = np.array(sims, dtype=np.float64)

    reliability_map = learn_reliability(idx_pairs, rows, profiles, lo_hi_map)
    for p in profiles:
        p.reliability_score = float(reliability_map.get(p.name, 1.0))

    composites = [max(1e-9, p.normalized_weight * p.reliability_score) for p in profiles]
    ssum = sum(composites)
    for p, c in zip(profiles, composites):
        p.composite_weight = c / ssum if ssum > 0 else 1.0 / len(profiles)

    plt.figure(figsize=(9, 4.5))
    plt.hist(sim_arr, bins=min(48, max(10, n_pairs // 25)), color="#4c72b0", edgecolor="white", alpha=0.85)
    plt.title("Distribution of weighted pairwise similarity (random pairs)")
    plt.xlabel("Similarity")
    plt.ylabel("Count")

    try:
        threshold, th_meta = kmeans_gap_threshold(sim_arr)
        plt.axvline(threshold, color="crimson", linewidth=2, label=f"Learned threshold = {threshold:.3f}")
        plt.legend()
        explanation = (
            "KMeans with k=2 was fit on 1D similarity scores from random row pairs. "
            "The two cluster centroids approximate 'clearly different' vs 'possibly same' groups; "
            "their midpoint is used as a data-driven match cutoff."
        )
        th_extra = {**th_meta, "method": explanation}
    except Exception as exc:  # noqa: BLE001
        threshold = float(np.median(sim_arr))
        th_extra = {
            "fallback": "median",
            "error": str(exc),
            "method": "Median similarity used because KMeans failed.",
        }
        plt.axvline(threshold, color="orange", linewidth=2, label=f"Fallback threshold = {threshold:.3f}")
        plt.legend()

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=120)
    plt.close()

    weights_payload = {
        "source_csv": str(csv_path),
        "n_rows": n_rows,
        "column_classification_path": str(cls_out.resolve()),
        "columns_excluded_from_learning": excluded_cols,
        "n_pairs_sampled": len(idx_pairs),
        "columns": {
            p.name: {
                "semantic_type": p.semantic_type,
                "physical_kind": p.physical_kind,
                "uniqueness_ratio": round(p.uniqueness_ratio, 6),
                "null_rate": round(p.null_rate, 6),
                "discriminative_score": round(p.discriminative_score, 6),
                "normalized_discriminative_weight": round(p.normalized_weight, 6),
                "reliability_score": round(p.reliability_score, 6),
                "composite_weight": round(p.composite_weight, 6),
            }
            for p in profiles
        },
        "notes": (
            "discriminative_score = uniqueness_ratio * (1 - null_rate); "
            "normalized_discriminative_weight sums to 1. composite_weight merges "
            "discriminative weight with empirical cross-field reliability."
        ),
    }
    write_json(OUT_WEIGHTS, weights_payload)

    threshold_payload = {
        "threshold": round(float(threshold), 6),
        "columns_excluded_from_learning": excluded_cols,
        "n_pairs": len(idx_pairs),
        "similarity_statistics": {
            "min": float(sim_arr.min()),
            "max": float(sim_arr.max()),
            "mean": float(sim_arr.mean()),
            "std": float(sim_arr.std(ddof=0)),
        },
        "plot_path": str(OUT_PLOT.resolve()),
        "n_rows": n_rows,
        **th_extra,
    }
    write_json(OUT_THRESHOLD, threshold_payload)

    write_markdown(
        csv_path,
        profiles,
        float(threshold),
        {**th_extra, "n_rows": n_rows},
        len(idx_pairs),
        excluded_cols,
    )

    print(f"Saved `{OUT_WEIGHTS}`")
    print(f"Saved `{OUT_THRESHOLD}`")
    print(f"Saved plot `{OUT_PLOT}`")
    print(f"Saved `{OUT_RULES}`")
    print("\nSimilarity preview (sample):")
    print(
        f"  min={sim_arr.min():.4f} max={sim_arr.max():.4f} "
        f"mean={sim_arr.mean():.4f} threshold={threshold:.4f}"
    )


def run_similarity_self_tests() -> None:
    """Standalone checks for char_jaccard and hybrid_field_similarity (no pytest)."""
    j_red_amber = char_jaccard("red", "amber")
    ok_jaccard = abs(j_red_amber - (1.0 / 3.0)) < 0.02
    print(
        f"1. char_jaccard('red', 'amber') ~0.33: "
        f"{'PASS' if ok_jaccard else 'FAIL'} (got {j_red_amber:.4f})"
    )

    emb_red = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    emb_amber = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    hybrid_score = hybrid_field_similarity("red", "amber", emb_red, emb_amber)
    ok_hybrid_differs = abs(hybrid_score - j_red_amber) > 1e-9
    print(
        f"2. hybrid_field_similarity != pure jaccard: "
        f"{'PASS' if ok_hybrid_differs else 'FAIL'} "
        f"(jaccard={j_red_amber:.4f}, hybrid={hybrid_score:.4f})"
    )

    empty_emb_i = _embedding_for_record({}, "red")
    empty_emb_j = _embedding_for_record({}, "amber")
    fallback_score = hybrid_field_similarity("red", "amber", empty_emb_i, empty_emb_j)
    ok_fallback = abs(fallback_score - j_red_amber) < 1e-9
    print(
        f"3. empty embeddings dict falls back to jaccard: "
        f"{'PASS' if ok_fallback else 'FAIL'} "
        f"(jaccard={j_red_amber:.4f}, fallback={fallback_score:.4f})"
    )

    # Test 4 — semantically similar words get higher hybrid than jaccard
    # Using embeddings from the actual global store if available
    emb_store = load_global_embeddings(GLOBAL_EMBEDDINGS_PATH)
    if emb_store:
        # Pick two records known to be the same entity but different descriptions
        # Load products_semantic.csv and find a pair with low jaccard but same MPN
        csv_path = PROJECT_ROOT / "data" / "products_semantic.csv"
        if csv_path.is_file():
            mpn_to_rows: dict[str, list[dict[str, str]]] = {}
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    mpn = row.get("manufacturer_part_number", "").strip().lower()
                    mpn_to_rows.setdefault(mpn, []).append(row)

            # Find a pair with same MPN, both have embeddings, descriptions differ
            test_pair = None
            for mpn, rows in mpn_to_rows.items():
                for i in range(len(rows)):
                    for j in range(i + 1, len(rows)):
                        pid_a = str(rows[i].get("product_id", "")).strip()
                        pid_b = str(rows[j].get("product_id", "")).strip()
                        da = rows[i].get("description", "").strip()
                        db = rows[j].get("description", "").strip()
                        if (
                            pid_a in emb_store
                            and pid_b in emb_store
                            and da
                            and db
                            and da.lower() != db.lower()
                        ):
                            jac = char_jaccard(da, db)
                            if jac < 0.5:  # lexically different
                                emb_a = _embedding_for_record(emb_store, pid_a)
                                emb_b = _embedding_for_record(emb_store, pid_b)
                                if emb_a is None or emb_b is None:
                                    continue
                                cos = cosine_similarity(emb_a, emb_b)
                                hybrid = hybrid_field_similarity(da, db, emb_a, emb_b)
                                test_pair = (da, db, jac, cos, hybrid)
                                break
                    if test_pair:
                        break
                if test_pair:
                    break

            if test_pair:
                da, db, jac, cos, hyb = test_pair
                ok = hyb > jac  # hybrid should beat jaccard for semantically similar text
                print(
                    f"4. Real embeddings: hybrid > jaccard for same-product descriptions: "
                    f"{'PASS' if ok else 'FAIL'}\n"
                    f"   '{da}'\n"
                    f"   '{db}'\n"
                    f"   jaccard={jac:.4f}  cosine={cos:.4f}  hybrid={hyb:.4f}"
                )
            else:
                print("4. Skipped — no qualifying pair found in embeddings store")
        else:
            print("4. Skipped — products_semantic.csv not found")
    else:
        print("4. Skipped — embeddings_global.pkl not found (run embed.py first)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Learn matching weights and threshold from a CSV."
    )
    ap.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=(
            "Input CSV path (absolute, or relative to project root). "
            f"Default: {DEFAULT_INPUT_PATH}"
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in similarity self-tests and exit.",
    )
    args = ap.parse_args()
    if args.self_test:
        run_similarity_self_tests()
    else:
        main(input_path=args.csv_path if args.csv_path is not None else DEFAULT_INPUT_PATH)
