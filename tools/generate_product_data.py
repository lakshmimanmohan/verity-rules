"""
Generate messy, duplicate-heavy product catalog CSV for MDM demos.
Uses controlled noise: typos, case drift, missing UPC/EAN, format drift.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import string
from pathlib import Path

from faker import Faker

SOURCES = ["CRM", "ERP", "Catalog"]
TOTAL_ROWS = 500
NUM_PRODUCTS = 200
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "products_demo.csv"

MANUFACTURER_CODES = [f"MFG-{i:02d}" for i in range(1, 21)]
BRAND_CODES = [f"BR-{i:02d}" for i in range(1, 16)]
MEMBER_IDS = [f"RET-{i:02d}" for i in range(1, 11)]


def typo_string(s: str, chance: float) -> str:
    if not s or random.random() > chance:
        return s
    chars = list(s)
    kind = random.choice(["swap", "drop", "dup", "sub"])
    idx = random.randrange(len(chars))
    if kind == "swap" and len(chars) > 1:
        j = min(idx + random.choice([-1, 1]), len(chars) - 1)
        chars[idx], chars[j] = chars[j], chars[idx]
    elif kind == "drop" and len(chars) > 1:
        chars.pop(idx)
    elif kind == "dup":
        chars.insert(idx, chars[idx])
    else:
        chars[idx] = random.choice(string.ascii_letters + string.digits)
    return "".join(chars)


def maybe_missing(value: str, prob: float) -> str:
    return "" if value and random.random() < prob else value


def mess_upc(raw: str) -> str:
    """Format drift for UPC/EAN (caller applies maybe_missing for ~15% null rate)."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 8:
        return raw
    roll = random.random()
    if roll < 0.22:
        return digits
    if roll < 0.42:
        return f"{digits[:1]} {digits[1:6]} {digits[6:]}" if len(digits) >= 7 else digits
    if roll < 0.58:
        return f"{digits[:6]}-{digits[6:]}" if len(digits) > 6 else digits
    if roll < 0.72:
        return digits.upper()
    if roll < 0.86:
        return f"EAN {digits}"
    return digits


def mess_manufacturer_part_number(mpn: str) -> str:
    roll = random.random()
    if roll < 0.14:
        return mpn.upper()
    if roll < 0.28:
        return mpn.lower()
    if roll < 0.36:
        return mpn.title()
    if roll < 0.44:
        parts = mpn.split("-")
        if len(parts) >= 2:
            i = random.randrange(len(parts))
            parts[i] = parts[i].swapcase()
            return "-".join(parts)
    if roll < 0.52:
        return typo_string(mpn, 1.0)
    return typo_string(mpn, 0.11)


def canonical_product(fake: Faker, entity_id: int) -> dict:
    """One golden product identity (shared across duplicate rows)."""
    mfg_code = MANUFACTURER_CODES[(entity_id - 1) % len(MANUFACTURER_CODES)]
    brand_code = BRAND_CODES[(entity_id * 7 + 3) % len(BRAND_CODES)]
    base_mpn = fake.bothify(text="???-#####-?", letters="ABCDEFGHJKLMNPQRSTUVWXYZ")
    digits = "".join(str(random.randint(0, 9)) for _ in range(12))
    return {
        "true_entity_id": entity_id,
        "manufacturer_part_number": base_mpn,
        "upc_ean": digits,
        "manufacturer_code": mfg_code,
        "brand_code": brand_code,
    }


def member_part_number(member_id: str, entity_id: int) -> str:
    """Retailer-assigned SKU: stable per retailer + canonical product."""
    return f"{member_id}-SKU-{entity_id:05d}"


def source_row(canonical: dict, fake: Faker, source_system: str, member_id: str) -> dict:
    te_id = canonical["true_entity_id"]
    mpn = mess_manufacturer_part_number(canonical["manufacturer_part_number"])
    mpn = typo_string(mpn, 0.06) if random.random() < 0.12 else mpn

    upc_raw = canonical["upc_ean"]
    upc_display = maybe_missing(mess_upc(upc_raw), 0.15) if upc_raw else ""

    msku = member_part_number(member_id, te_id)
    if random.random() < 0.08:
        msku = typo_string(msku, 0.35)

    return {
        "product_id": str(fake.uuid4()),
        "manufacturer_part_number": mpn,
        "member_part_number": msku,
        "upc_ean": upc_display,
        "manufacturer_code": canonical["manufacturer_code"],
        "brand_code": canonical["brand_code"],
        "member_id": member_id,
        "source_system": source_system,
        "true_entity_id": te_id,
    }


def row_counts_for_entities(num_entities: int, total_rows: int) -> list[int]:
    counts = [random.randint(2, 3) for _ in range(num_entities)]
    s = sum(counts)
    while s < total_rows:
        counts[random.randrange(num_entities)] += 1
        s += 1
    while s > total_rows:
        idx_candidates = [i for i, c in enumerate(counts) if c > 2]
        if not idx_candidates:
            idx_candidates = [i for i, c in enumerate(counts) if c > 1]
        if not idx_candidates:
            raise RuntimeError("Cannot reduce row counts safely")
        i = random.choice(idx_candidates)
        counts[i] -= 1
        s -= 1
    return counts


def main(output_path: Path | None = None) -> None:
    random.seed(42)
    Faker.seed(42)
    fake = Faker("en_US")

    path = output_path if output_path is not None else OUTPUT_PATH

    canon = [canonical_product(fake, i + 1) for i in range(NUM_PRODUCTS)]
    counts = row_counts_for_entities(NUM_PRODUCTS, TOTAL_ROWS)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "product_id",
        "manufacturer_part_number",
        "member_part_number",
        "upc_ean",
        "manufacturer_code",
        "brand_code",
        "member_id",
        "source_system",
        "true_entity_id",
    ]

    rows: list[dict] = []
    source_cycle = ["CRM", "ERP", "Catalog"]
    for base, k in zip(canon, counts):
        for i in range(k):
            if i < len(source_cycle):
                src = source_cycle[i]
            else:
                src = random.choice(SOURCES)
            member_id = random.choice(MEMBER_IDS)
            rows.append(source_row(base, fake, src, member_id))

    random.shuffle(rows)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} product rows ({NUM_PRODUCTS} canonical products) saved to {path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate messy product catalog CSV for MDM demos."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Output CSV file (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()
    main(output_path=args.output)
