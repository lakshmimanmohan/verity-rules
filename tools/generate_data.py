"""
Generate messy, duplicate-heavy customer CSV for MDM evaluation.
Uses Faker with controlled noise: typos, missing fields, format drift.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import string
from pathlib import Path

from faker import Faker

SOURCES = ["CRM", "ERP", "Billing"]
TOTAL_ROWS = 4000
NUM_ENTITIES = 2000
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "customers_raw.csv"


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
        chars[idx] = random.choice(string.ascii_letters)
    return "".join(chars)


def maybe_missing(value: str, prob: float) -> str:
    return "" if value and random.random() < prob else value


def mess_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if not domain:
        return email
    roll = random.random()
    if roll < 0.2:
        return f"{local}{random.randint(0, 99)}@{domain}"
    if roll < 0.35:
        alt = random.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com"])
        return f"{local}@{alt}"
    if roll < 0.5:
        return email.upper()
    if roll < 0.65:
        return email.replace("@", "(at)")
    return email


def mess_phone(raw: str) -> str | None:
    """Return noisy phone or None meaning omit field."""
    if random.random() < 0.12:
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 10:
        return raw
    roll = random.random()
    if roll < 0.25:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:10]}"
    if roll < 0.45:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:10]}"
    if roll < 0.65:
        return f"+1.{digits[:3]}.{digits[3:6]}.{digits[6:10]}"
    if roll < 0.8:
        return digits
    return f"+1 {digits[:3]} {digits[3:6]} {digits[6:10]}"


def mess_address_line(addr: str) -> str:
    mapping = {
        "Street": "St",
        "Road": "Rd",
        "Avenue": "Ave",
        "Boulevard": "Blvd",
        "Drive": "Dr",
        "Lane": "Ln",
        "Court": "Ct",
    }
    out = addr
    for long, short in mapping.items():
        if long in out and random.random() < 0.35:
            out = out.replace(long, short, 1)
    if random.random() < 0.15:
        out = typo_string(out, 1.0)
    return out


def mess_city(city: str) -> str:
    if random.random() < 0.2:
        return city.upper()
    if random.random() < 0.15:
        return typo_string(city, 1.0)
    return city


def zip_variants(zip_code: str) -> str:
    if "-" in zip_code:
        zip_code = zip_code.split("-", 1)[0]
    if random.random() < 0.25 and len(zip_code) == 5:
        zip_code = zip_code[:3] + " " + zip_code[3:]
    if random.random() < 0.1:
        zip_code += random.choice(["", " USA", "-0000"])
    return zip_code


def canonical_profile(fake: Faker, entity_id: int) -> dict:
    name = fake.name().split(maxsplit=1)
    first = name[0] if len(name) > 0 else fake.first_name()
    last = name[1] if len(name) > 1 else fake.last_name()
    return {
       
        "first_name": first,
        "last_name": last,
        "email": fake.email(),
        "phone": fake.phone_number(),
        "address": fake.street_address(),
        "city": fake.city(),
        "zip_code": fake.zipcode(),
        "ssn": f"{random.randint(100, 899):03d}-{random.randint(10, 99):02d}-{random.randint(1000, 9999):04d}",
    }


def source_row(canonical: dict, fake: Faker) -> dict:
    cid = fake.uuid4()
    fn = maybe_missing(typo_string(canonical["first_name"], 0.18), 0.02)
    ln = maybe_missing(typo_string(canonical["last_name"], 0.18), 0.02)
    email_raw = canonical["email"]
    if random.random() < 0.08:
        base = fn.lower().replace(".", "") if fn else canonical["email"].split("@")[0]
        email_raw = f"{base}{random.randint(1, 9)}@{email_raw.split('@')[-1]}"
    email = maybe_missing(mess_email(email_raw), 0.04)
    phone_out = mess_phone(canonical["phone"]) if canonical["phone"] else None
    if phone_out is None:
        phone_display = ""
    else:
        phone_display = typo_string(str(phone_out), 0.05) if random.random() < 0.08 else phone_out

    addr = mess_address_line(canonical["address"])
    city = mess_city(canonical["city"])
    zc = zip_variants(canonical["zip_code"])
    addr = typo_string(addr, 0.06) if random.random() < 0.25 else addr

    ssn = maybe_missing(typo_string(canonical["ssn"], 0.15), 0.05)

    return {
        "customer_id": cid,
        "first_name": fn,
        "last_name": ln,
        "email": email,
        "phone": phone_display,
        "address": addr,
        "city": city,
        "zip_code": zc,
        "SSN": ssn,
        "source_system": random.choice(SOURCES)
      
    }


def row_counts_for_entities(num_entities: int, total_rows: int) -> list[int]:
    """
    Assign each entity 1–3 variants first, then add rows at random entities
    until total_rows is reached (so we can reach 500 while staying grounded
    in 100 distinct people).
    """
    counts = [random.randint(1, 3) for _ in range(num_entities)]
    s = sum(counts)
    while s < total_rows:
        counts[random.randrange(num_entities)] += 1
        s += 1
    while s > total_rows:
        # shrink from entities with extras beyond 1
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

    canon = [canonical_profile(fake, i + 1) for i in range(NUM_ENTITIES)]
    counts = row_counts_for_entities(NUM_ENTITIES, TOTAL_ROWS)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "customer_id",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "city",
        "zip_code",
        "SSN",
        "source_system",
        "true_entity_id",
    ]

    rows: list[dict] = []
    for entity_id, (base, k) in enumerate(zip(canon, counts), start=1):
        for _ in range(k):
            row = source_row(base, fake)
            row["true_entity_id"] = entity_id
            rows.append(row)
    random.shuffle(rows)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} records saved to {path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate messy, duplicate-heavy customer CSV for MDM evaluation."
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