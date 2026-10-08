"""
Print every column ranked by composite_weight with reliability, anchor count and reason.
Does not read truth files.

Usage: python tools/report_ranked_columns.py data/customers.csv data/products.csv [--out FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verity_rules.core import learn_rules  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out: list[str] = []
    for path in a.csv:
        df = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", None)
        res = learn_rules(df.to_dict(orient="records"), debug=True)
        dbg = res["reliability_debug"]
        out += ["=" * 100, f"{path}  records={res['n_records']}  separability "
                f"gap {res['separability']['gap']}", ""]
        ranked = sorted(res["columns"].items(), key=lambda kv: -kv[1]["composite_weight"])
        for rank, (name, c) in enumerate(ranked, 1):
            out += [f"{rank}. {name}  [{c['semantic_type']}, {c['comparison_method']}]",
                    f"   composite_weight={c['composite_weight']:.6f}  reliability={c['reliability_score']:.6f}  "
                    f"anchors={dbg['anchor_count'][name]} (min_agree={dbg['min_agree_used'][name]})",
                    f"   {c['reason']}"]
        out += ["", "Notes:"] + [f"  {n}" for n in res["notes"]] + [""]
    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
