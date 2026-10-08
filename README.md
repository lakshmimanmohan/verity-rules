# Verity Rules

Verity reads a CSV with any schema and learns which columns identify an entity and how much each one should count. It outputs rules you can use in any matching tool. It does not match, merge or clean records.

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

If PowerShell blocks activation, skip it and call `.venv\Scripts\python.exe` and `.venv\Scripts\verity.exe` directly.

## Usage

```powershell
verity learn data\customers.csv --out out\customers
verity learn data\products.csv --out out\products
```

Also works as `python -m verity_rules learn ...`.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | required | Output folder, created if missing |
| `--seed N` | 42 | Random seed. Same input and seed give the same output |
| `--max-rows N` | 20000 | Larger files are sampled down to N rows (fixed seed) and the original row count is noted |
| `--mask-columns A,B` | none | Replace the values of these columns with `***` in `calibration_pairs.csv` and in the pairs in `rules.json` (empty values stay empty) |
| `--no-pairs` | off | Write no pairs files and leave `calibration_pairs` empty in `rules.json` |
| `--debug` | off | Also write `debug.json` (internal scores and threshold; not part of the public rules) |

## Output

| File | Contents |
|---|---|
| `rules.json` | Learned rules: per-column weight, reliability, comparison method and reason, plus expected pair coverage and calibration pairs |
| `rules_report.md` | Plain-language report of the rules, for people new to Verity |
| `calibration_pairs.csv` | Sample pairs to label by hand, in random order and without Verity's labels. Fill in `your_label` |
| `calibration_pairs_key.csv` | `pair_id` and `suggested_label` (Verity's guess) for each pair. Open it after you label |

`expected_pair_coverage` is the sum over kept columns of weight x (1 - null_rate)^2, the chance that both records of a pair have a value in that column. Low values mean many pairs share few columns.

Set your own match cutoff in your own tool by scoring `calibration_pairs.csv` there. Verity does not return one.

The pairs files and the pairs inside `rules.json` contain values from the source data. Handle them like the source data, or use `--mask-columns` or `--no-pairs`.

## What Verity excludes and why

No field mapping is needed. Verity decides from the data which columns describe an entity and which are system or bookkeeping columns. Column names are one signal among others (uniqueness, number of distinct values, what the values look like). The name words below are generic and are not tied to any dataset. This list is read from `verity_rules/column_classification.py`.

A column is excluded if any of these apply.

**Name signals**

- Reserved names: starts with `_`, ends with `_truth` or `_gold`, or is exactly `label`, `labels`, `y`, `target` or `gold_label`.
- System or process words that match anywhere in the name: `category`, `created`, `updated`, `timestamp`, `modified`, `score`.
- System or process words that match only as a whole word, not next to a letter, digit or underscore: `source`, `system`, `status`, `flag`, `type`, `tier`, `segment`, `class`, `date`, `rank`. So a column named `type` matches, `customer_type` does not.
- `id`, `key` or `code` as a whole underscore-separated part of the name (for example `customer_id`), only when every row has a different value.
- Flag words anywhere in the name: `is_`, `has_`, `active`, `enabled`, `verified`, `opted`, only when the column has two or three distinct values.
- Record id: a column named `id`, `record_id`, `uuid` or `key`, or ending in `_id` or `_key`, is chosen as the record id and left out of the rules. If none match, Verity falls back to a near-unique column.

**Value signals (no name involved)**

- 99% or more of the values are different (primary key).
- Every record has the same value.
- Fewer than 5 distinct values, or fewer than 10 distinct values and under 5% unique.
- 72% or more of the values look like generated labels (`WORD_123`, or starting `UNKNOWN_`).

The report lists each excluded column with its reason.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```
