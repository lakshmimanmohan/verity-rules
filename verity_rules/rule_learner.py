"""
Discover schema, discriminative weights, similarity threshold, and field reliability
from an arbitrary CSV (no assumed column names).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


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
    inferred_kind: str = ""  # physical kind before the IDENTIFIER override; "" means physical_kind


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
    """uniqueness_ratio = distinct non-null values / non-null count; null_rate = nulls / n_rows."""
    nulls = sum(1 for v in values if is_null(v))
    null_rate = nulls / n_rows if n_rows else 0.0
    non_null = n_rows - nulls
    uniq = len({token_key(v) for v in values if not is_null(v)})
    uniqueness_ratio = uniq / non_null if non_null else 0.0
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
        inferred = pk
        sem = infer_semantic_type(h, colvals, uq, pk, is_metadata=h in meta)
        if str(sem).strip().upper() == "IDENTIFIER":
            pk = "STRING"
        disc = uq * (1.0 - nr)
        profiles.append(
            ColumnProfile(
                name=h,
                semantic_type=sem,
                physical_kind=pk,
                inferred_kind=inferred,
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


COMPARATOR_VARIANTS = ("v0", "v1", "v2", "v3", "v4")


def check_comparator_variant(variant: str) -> str:
    if variant not in COMPARATOR_VARIANTS:
        raise ValueError(f"comparator_variant must be one of {COMPARATOR_VARIANTS}, got {variant!r}")
    return variant


def _alnum_norm(v: Any) -> str:
    """Lowercase, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", clean_str(v).lower())


def _bigrams(s: str) -> set[str]:
    """Adjacent character pairs; a one-character string is its own single gram."""
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())


def bigram_jaccard(a: Any, b: Any) -> float:
    """Jaccard similarity of the bigram sets of the alphanumeric-normalized values.
    Two values that normalize to nothing are equal only if their stripped text is equal."""
    na, nb = _alnum_norm(a), _alnum_norm(b)
    if not na and not nb:
        return 1.0 if clean_str(a) == clean_str(b) else 0.0
    ga, gb = _bigrams(na), _bigrams(nb)
    return len(ga & gb) / len(ga | gb)


def identifier_similarity(va: Any, vb: Any, comparator_variant: str) -> float:
    """IDENTIFIER columns that are not email or date, by variant (v1 exact, v2 normalized exact, v3+ bigram)."""
    if comparator_variant in ("v0", "v1"):
        return 1.0 if clean_str(va) == clean_str(vb) else 0.0
    if comparator_variant == "v2":
        na, nb = _alnum_norm(va), _alnum_norm(vb)
        if not na and not nb:
            return 1.0 if clean_str(va) == clean_str(vb) else 0.0
        return 1.0 if na == nb else 0.0
    return bigram_jaccard(va, vb)


def field_similarity(
    va: Any,
    vb: Any,
    prof: ColumnProfile,
    lo_hi: tuple[float | None, float | None],
    *,
    record_emb_i: np.ndarray | None = None,
    record_emb_j: np.ndarray | None = None,
    comparator_variant: str = "v4",
) -> float:
    if is_null(va) or is_null(vb):
        return 0.0
    semantic_type = str(prof.semantic_type).strip().upper()
    physical_type = str(prof.physical_kind).strip().upper()
    if semantic_type == "IDENTIFIER":
        if comparator_variant == "v0":
            return 1.0 if clean_str(va) == clean_str(vb) else 0.0
        # v1+: email and date columns keep their own comparator even when classed IDENTIFIER.
        detected = str(prof.inferred_kind or prof.physical_kind).strip().upper()
        if detected == "EMAIL":
            return email_similarity(va, vb)
        if detected == "DATE":
            return date_similarity(va, vb)
        return identifier_similarity(va, vb, comparator_variant)
    if physical_type == "EMAIL":
        return email_similarity(va, vb)
    if physical_type == "DATE":
        return date_similarity(va, vb)
    if physical_type == "NUMBER":
        lo, hi = lo_hi
        if lo is None or hi is None:
            return char_jaccard(clean_str(va), clean_str(vb))
        return numeric_similarity(va, vb, lo, hi)
    if comparator_variant == "v4" and (record_emb_i is None or record_emb_j is None):
        # text_character columns (also the hybrid path when no embedding is available)
        return bigram_jaccard(va, vb)
    if semantic_type in ("DESCRIPTIVE", "NAME", "STREET_ADDRESS") and physical_type == "STRING":
        return hybrid_field_similarity(va, vb, record_emb_i, record_emb_j)
    return char_jaccard(clean_str(va), clean_str(vb))


def comparison_method(prof: ColumnProfile, embeddings_used: bool, comparator_variant: str = "v4") -> str:
    """Name the comparison field_similarity applies to this column. Keep in sync with it."""
    semantic_type = str(prof.semantic_type).strip().upper()
    physical_type = str(prof.physical_kind).strip().upper()
    text = "text_bigram" if comparator_variant == "v4" else "text_character"
    if semantic_type == "IDENTIFIER":
        if comparator_variant == "v0":
            return "exact"
        detected = str(prof.inferred_kind or prof.physical_kind).strip().upper()
        if detected == "EMAIL":
            return "email"
        if detected == "DATE":
            return "date"
        return {"v1": "exact", "v2": "exact_normalized"}.get(comparator_variant, "bigram_jaccard")
    if physical_type == "EMAIL":
        return "email"
    if physical_type == "DATE":
        return "date"
    if physical_type == "NUMBER":
        return "numeric_range"
    if semantic_type in ("DESCRIPTIVE", "NAME", "STREET_ADDRESS") and physical_type == "STRING":
        return "text_hybrid" if embeddings_used else text
    return text


def weighted_pair_similarity(
    row_i: dict[str, Any],
    row_j: dict[str, Any],
    profiles: list[ColumnProfile],
    lo_hi_map: dict[str, tuple[float | None, float | None]],
    embeddings: dict[str, list[float]] | None = None,
    row_ids: tuple[str, str] | None = None,
    weight_attr: str = "normalized_weight",
    comparator_variant: str = "v4",
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
        w = getattr(p, weight_attr)
        s = field_similarity(
            vi,
            vj,
            p,
            lo_hi_map[p.name],
            record_emb_i=record_emb_i,
            record_emb_j=record_emb_j,
            comparator_variant=comparator_variant,
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


ANCHOR_MIN_UNIQUENESS = 0.5
ANCHOR_MIN_COUNT = 30
RANDOM_MEAN_CEILING = 0.999


def _alnum_key(v: Any) -> str:
    """Lowercase, alphanumerics only. Empty string when null or nothing is left."""
    if is_null(v):
        return ""
    return re.sub(r"[^a-z0-9]", "", clean_str(v).lower())


def learn_reliability_v2(
    candidate_pairs: list[tuple[int, int]],
    rows: list[dict[str, Any]],
    profiles: list[ColumnProfile],
    lo_hi_map: dict[str, tuple[float | None, float | None]],
    *,
    min_agree: int = 2,
    rescale: bool = False,
    relax: bool = True,
    embeddings: dict[str, list[float]] | None = None,
    row_ids: list[str] | None = None,
    random_pairs: list[tuple[int, int]] | None = None,
    comparator_variant: str = "v4",
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Chance-corrected reliability: how much more similar a column is on anchor pairs
    than on random pairs.

        reliability = clip((anchor_mean - random_mean) / (1 - random_mean), 0, 1)

    anchor_mean is the mean field_similarity over the column's anchors; random_mean is
    the mean over `random_pairs` where both values are non-null. If random_mean is
    RANDOM_MEAN_CEILING or more the column agrees on almost any pair, so reliability is 0.
    With random_pairs=None the correction is off (random_mean = 0, the raw anchor mean).
    If every column with enough anchors ends at 0, all reliabilities are set to 1.0
    (uniqueness-only weights) and a note says so.

    An anchor for column C is a candidate pair that agrees (after lowercasing and
    keeping only alphanumerics) on a value in at least `min_agree` columns other
    than C, among columns with uniqueness_ratio >= ANCHOR_MIN_UNIQUENESS, and where
    both values of C are non-null. A column with fewer than ANCHOR_MIN_COUNT
    anchors is relaxed one level at a time down to 1 (when `relax`) with a note;
    if still under ANCHOR_MIN_COUNT it gets 1.0 and a note. `rescale` applies the
    0.1-1.0 squash over the columns that had enough anchors.

    Returns (reliability, info). info has anchor_count, min_agree_used and
    anchor_pairs (column -> list of (i, j)) at the level used, anchor_mean and
    random_mean (per column), raw (the chance-corrected value before rescale) and notes.
    """
    anchor_cols = [p for p in profiles if p.uniqueness_ratio >= ANCHOR_MIN_UNIQUENESS]
    keys = {p.name: [_alnum_key(r.get(p.name)) for r in rows] for p in anchor_cols}

    # column -> [(agreeing other columns, i, j, similarity)]
    obs: dict[str, list[tuple[int, int, int, float]]] = {p.name: [] for p in profiles}

    for i, j in candidate_pairs:
        agree = {p.name for p in anchor_cols
                 if keys[p.name][i] and keys[p.name][i] == keys[p.name][j]}
        if not agree:
            continue
        emb_i = emb_j = None
        if embeddings and row_ids is not None:
            emb_i = _embedding_for_record(embeddings, row_ids[i])
            emb_j = _embedding_for_record(embeddings, row_ids[j])
        ri, rj = rows[i], rows[j]
        for p in profiles:
            k = len(agree - {p.name})
            if k < 1:
                continue
            vi, vj = ri.get(p.name), rj.get(p.name)
            if is_null(vi) or is_null(vj):
                continue
            obs[p.name].append((k, i, j, field_similarity(
                vi, vj, p, lo_hi_map[p.name], record_emb_i=emb_i, record_emb_j=emb_j,
                comparator_variant=comparator_variant)))

    notes: list[str] = []
    anchor_pairs: dict[str, list[tuple[int, int]]] = {}
    used: dict[str, int] = {}
    raw: dict[str, float] = {}
    lowest = 1 if relax else min_agree
    for p in profiles:
        level = min_agree
        while True:
            sel = [o for o in obs[p.name] if o[0] >= level]
            if len(sel) >= ANCHOR_MIN_COUNT or level <= lowest:
                break
            level -= 1
        if level < min_agree and len(sel) >= ANCHOR_MIN_COUNT:
            notes.append(f"{p.name}: anchor rule relaxed from {min_agree} to {level} "
                         f"agreeing columns ({len(sel)} anchor pairs).")
        used[p.name] = level
        anchor_pairs[p.name] = [(o[1], o[2]) for o in sel]
        if len(sel) >= ANCHOR_MIN_COUNT:
            raw[p.name] = float(sum(o[3] for o in sel) / len(sel))
        else:
            notes.append(f"{p.name}: only {len(sel)} anchor pairs (need {ANCHOR_MIN_COUNT}) "
                         f"even with anchor rule relaxed; reliability set to 1.0."
                         if level < min_agree else
                         f"{p.name}: only {len(sel)} anchor pairs (need {ANCHOR_MIN_COUNT}); "
                         "reliability set to 1.0.")

    random_mean: dict[str, float] = {}
    anchor_mean = dict(raw)
    if random_pairs:
        rs: dict[str, list[float]] = {p.name: [] for p in profiles}
        for i, j in random_pairs:
            emb_i = emb_j = None
            if embeddings and row_ids is not None:
                emb_i = _embedding_for_record(embeddings, row_ids[i])
                emb_j = _embedding_for_record(embeddings, row_ids[j])
            ri, rj = rows[i], rows[j]
            for p in profiles:
                vi, vj = ri.get(p.name), rj.get(p.name)
                if is_null(vi) or is_null(vj):
                    continue
                rs[p.name].append(field_similarity(
                    vi, vj, p, lo_hi_map[p.name], record_emb_i=emb_i, record_emb_j=emb_j,
                    comparator_variant=comparator_variant))
        random_mean = {k: float(sum(v) / len(v)) for k, v in rs.items() if v}
    for name in list(raw):
        r = random_mean.get(name, 0.0)
        if r >= RANDOM_MEAN_CEILING:
            raw[name] = 0.0
            notes.append(f"{name}: values already agree on random pairs (mean similarity {r:.3f}); "
                         "reliability set to 0.")
        else:
            raw[name] = float(min(1.0, max(0.0, (anchor_mean[name] - r) / (1.0 - r))))
    if raw and all(v == 0.0 for v in raw.values()):
        raw = {}
        notes.append("Every column's similarity on likely duplicates is no better than on random "
                     "pairs; reliability set to 1.0 for all columns, so weights come from "
                     "uniqueness only.")

    reliability = {p.name: raw.get(p.name, 1.0) for p in profiles}
    if rescale and raw:
        lo, hi = min(raw.values()), max(raw.values())
        for k in raw:
            reliability[k] = 0.1 + 0.9 * (raw[k] - lo) / (hi - lo) if hi > lo else 1.0
    info = {"anchor_count": {k: len(v) for k, v in anchor_pairs.items()},
            "min_agree_used": used, "anchor_pairs": anchor_pairs,
            "anchor_mean": anchor_mean, "random_mean": random_mean,
            "raw": raw, "notes": notes}
    return reliability, info


def kmeans_gap_threshold(sim_array: np.ndarray) -> tuple[float, dict[str, float]]:
    from sklearn.cluster import KMeans  # optional (extra "debug"); only the debug threshold uses it
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
