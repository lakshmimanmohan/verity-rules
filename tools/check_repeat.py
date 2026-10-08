"""Run learn_rules twice on a CSV and say whether the output is identical."""
import csv
import json
import sys

from verity_rules.core import learn_rules

path = sys.argv[1]
with open(path, newline="", encoding="utf-8-sig") as f:
    records = list(csv.DictReader(f))

a = json.dumps(learn_rules(records), sort_keys=True)
b = json.dumps(learn_rules(records), sort_keys=True)
print(f"{path}: {len(records)} records, run 1 and run 2 are",
      "IDENTICAL" if a == b else "DIFFERENT")