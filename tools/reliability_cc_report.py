r"""
Report for the uniqueness and chance-corrected-reliability changes. Not shipped to users.
Reads validation/baseline (before both changes) and validation/after_uniqueness (after the first),
runs the current code, and writes validation/reliability_cc_results.md. Report only, no tuning.

  .venv\Scripts\python.exe tools/reliability_cc_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import Dataset, auc, table  # noqa: E402
from verity_rules.core import learn_rules  # noqa: E402

BASE, UNIQ, VAL = ROOT / "validation" / "baseline", ROOT / "validation" / "after_uniqueness", ROOT / "validation"
DATA = ROOT / "data"
SEED = 42


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: float) -> str:
    return f"{x:.4f}"


def uniq_section(name: str, old: dict, new: dict) -> list[str]:
    rows = [[c, f(old["columns"][c]["uniqueness_ratio"]), f(new["columns"][c]["uniqueness_ratio"]),
             f(old["columns"][c]["composite_weight"]), f(new["columns"][c]["composite_weight"]),
             f"{new['columns'][c]['composite_weight'] - old['columns'][c]['composite_weight']:+.4f}"]
            for c in new["columns"]]
    return [f"### {name}", ""] + table(
        ["Column", "Uniqueness old", "Uniqueness new", "Weight old", "Weight new", "Change"], rows) + [
        f"Candidate pairs: {old['run']['n_candidate_pairs']} -> {new['run']['n_candidate_pairs']}. "
        f"Separability gap: {old['separability']['gap']} -> {new['separability']['gap']}.", ""]


def rel_section(name: str, old: dict, new: dict) -> list[str]:
    rows = [[c, f(old["columns"][c]["reliability_score"]), f(new["columns"][c]["reliability_score"]),
             f(old["columns"][c]["composite_weight"]), f(new["columns"][c]["composite_weight"]),
             f"{new['columns'][c]['composite_weight'] - old['columns'][c]['composite_weight']:+.4f}"]
            for c in new["columns"]]
    return [f"### {name}", ""] + table(
        ["Column", "Reliability before", "Reliability after", "Weight before", "Weight after", "Change"], rows) + [
        f"Separability gap: {old['separability']['gap']} -> {new['separability']['gap']}.", ""]


def compare(name: str, csv: Path, old: dict) -> list[str]:
    """Old vs new per-column weights and hard-negative AUC for one dataset."""
    ds = Dataset(name, csv, DATA / "customers_truth.csv")
    new = ds.result
    cols = ds.columns
    v1 = {c: v["composite_weight"] for c, v in learn_rules(ds.records, seed=SEED, reliability="v1")["columns"].items()}
    learned = {c: new["columns"][c]["composite_weight"] for c in cols}
    old_w = {c: old["columns"].get(c, {}).get("composite_weight", 0.0) for c in cols}
    schemes = ds.weight_schemes(learned, v1)
    neg, n_cand, n_all = ds.hard_negatives()
    a = {"learned (new)": auc(ds, learned, neg), "learned (old code)": auc(ds, old_w, neg),
         "equal": auc(ds, schemes["equal"], neg), "uniqueness only": auc(ds, schemes["uniqueness"], neg),
         "v1": auc(ds, schemes["v1"], neg)}
    L = [f"### {name}", "",
         f"Truth: {len(ds.dup_pairs)} true duplicate pairs. Hard negatives (candidate pairs that are not true "
         f"duplicates, built with the new uniqueness): {len(neg)} of {n_cand} candidate pairs"
         + (f" (capped from {n_all})" if n_all > len(neg) else "") + ". "
         f"Kept columns: {', '.join(cols)}. Excluded: {', '.join(e['name'] for e in new['columns_excluded'])}. "
         f"Separability: {new['separability']['label']} (gap {new['separability']['gap']}).", ""]
    rows = []
    for c in cols:
        o, n_ = old["columns"].get(c), new["columns"][c]
        rows.append([c, f(o["uniqueness_ratio"]) if o else "-", f(n_["uniqueness_ratio"]), f(n_["null_rate"]),
                     f(o["reliability_score"]) if o else "-", f(n_["reliability_score"]),
                     f(old_w[c]), f(learned[c]), f"{learned[c] - old_w[c]:+.4f}"])
    L += table(["Column", "Uniqueness old", "new", "Null rate", "Reliability old", "new",
                "Weight old", "Weight new", "Change"], rows)
    L += ["AUC against hard negatives (same positives and negatives for every row):", ""]
    L += table(["Weighting", "AUC"], [[k, f"{v:.6f}"] for k, v in a.items()])
    L += ["Weights used by each scheme:", ""]
    L += table(["Column", "learned (new)", "learned (old code)", "equal", "uniqueness only", "v1"],
               [[c, f"{learned[c]:.4f}", f"{old_w[c]:.4f}", f"{schemes['equal'][c]:.4f}",
                 f"{schemes['uniqueness'][c]:.4f}", f"{schemes['v1'][c]:.4f}"] for c in cols])
    return L


def main() -> int:
    L = ["# Uniqueness and chance-corrected reliability: results", "",
         "Report only, no tuning. Old = code at commit b2436de (before both changes). Seed 42.", "",
         "## Change 1: uniqueness over non-null values", ""]
    for n in ("customers", "products", "degraded_C"):
        L += uniq_section(n if n != "degraded_C" else "Copy C (blank 50% of every column)",
                          load(BASE / f"{n}_rules.json"), load(UNIQ / f"{n}_rules.json"))
    c_old, c_new = load(BASE / "degraded_C_rules.json"), load(UNIQ / "degraded_C_rules.json")
    L += [f"Copy C candidate pairs: {c_old['run']['n_candidate_pairs']} before, "
          f"{c_new['run']['n_candidate_pairs']} after, so candidates now exist.", "",
          "## Change 2: chance-corrected reliability (before = after change 1 only)", ""]
    for n in ("customers", "products", "degraded_C"):
        L += rel_section(n if n != "degraded_C" else "Copy C", load(UNIQ / f"{n}_rules.json"),
                         load(VAL / "after_reliability" / f"{n}_rules.json"))
    L += ["## Copy H and the original customers: old code vs both changes", "",
          "Copy H = customers with 50% of email and phone blanked, typos in 30% of first_name, last_name "
          "and address, 60% of zip_code and city scrambled with values from other rows, and 2 noise columns "
          "of 8-letter words and 2 of 10-digit strings (`tools/degrade.py`, seed 42). Truth file unchanged. "
          "Built with: `python tools/degrade.py data/customers.csv data/customers_degraded_H.csv --blank email,phone 0.5 "
          "--typo first_name,last_name,address 0.3 --scramble zip_code,city 0.6 --noise-cols 2`. "
          "The four noise columns are 100% unique, so Verity excludes them as primary keys and they get no weight.", ""]
    L += compare("customers (original)", DATA / "customers.csv", load(BASE / "customers_rules.json"))
    L += compare("Copy H", DATA / "customers_degraded_H.csv", load(BASE / "degraded_H_rules.json"))
    out = VAL / "reliability_cc_results.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
