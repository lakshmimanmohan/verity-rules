"""rules.json must be strict JSON: missing values are null, never NaN or Infinity."""
import json
from pathlib import Path

import pytest

from verity_rules.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["customers", "products"]


def _reject(token):
    raise ValueError(f"non-standard JSON constant {token}")


@pytest.mark.parametrize("name", DATASETS)
def test_rules_json_has_no_nan_or_infinity(name, tmp_path):
    src = ROOT / "data" / f"{name}.csv"
    if not src.is_file():
        pytest.skip(f"{src} not available (data/ is git-ignored)")
    assert main(["learn", str(src), "--out", str(tmp_path)]) == 0
    text = (tmp_path / "rules.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=_reject)


def test_rules_json_nulls_for_missing_values(tmp_path):
    src = tmp_path / "in.csv"
    rows = ["id,name,score"] + [f"{i},{'Name' + str(i % 7)},{'' if i % 3 == 0 else i * 1.5}"
                                for i in range(60)]
    src.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert main(["learn", str(src), "--out", str(tmp_path / "o")]) == 0
    json.loads((tmp_path / "o" / "rules.json").read_text(encoding="utf-8"),
               parse_constant=_reject)
