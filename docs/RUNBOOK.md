# Verity runbook

Verity reads a CSV and learns which columns identify an entity and how much each should count. It writes rules (weights, comparison methods, reasons) and sample pairs to label. It does not match, merge or clean records, and it returns no match cutoff.

## When to use Verity, and when not

Use it when you have a table that probably holds duplicates and you need to know which columns to compare and how heavily, before you configure a matching tool. Do not use it to find or merge duplicates, to clean data, or to get a ready cutoff. It needs at least 2 rows; if the file has no unique id column, row numbers are used. With no or few shared values in distinctive columns it finds no candidate pairs and falls back to uniqueness-only weights (the run prints a warning).

## Install

Needs Python 3.10 or newer. Wheel location: <ADD PATH>

In a fresh virtual environment:

```
pip install verity_rules-0.1.2-py3-none-any.whl
verity --version
```

`pip install "verity_rules-0.1.2-py3-none-any.whl[debug]"` adds scikit-learn, used only for the debug threshold in `--debug`.

## Run

```
verity learn customers.csv --out out/customers
```

It writes four files to the folder: `rules.json`, `rules_report.md`, `calibration_pairs.csv`, `calibration_pairs_key.csv`. Same file and seed give the same output.

Run time and memory: about 1 minute and 200 MB for 20,000 rows; about 90 seconds and 350 MB for 100,000 rows with `--max-rows 100000`.

### The record id column

A record id has no blank values and at least 99 percent distinct values. Verity picks one that qualifies, preferring names such as `id`, `record_id`, `uuid`, `key` or ending in `_id` or `_key`, and leaves it out of the weights. The pairs files show its values.

If no column qualifies, Verity does not pick one. It adds a row-number column named `_row` (1-based position in the file after the header; `_row_2` and so on if the file already has `_row`) and uses it as the record id. The run prints a warning, and `rules.json` notes and the report say: "No unique id column was found. Row numbers (_row) are used as record ids in the pairs files." Every other column, such as email, stays in the weights.

If Verity picks a column whose name does not read as an id (it is not `id`, `record_id`, `uuid` or `key`, and does not end in `_id` or `_key`), the run prints a warning, and `rules.json` notes and the report say: "Using <col> as the record id because it has no blanks and every value is distinct. If this is not an id, use --id-column or add row numbers." Nothing is printed when the name reads as an id or when you pass `--id-column`.

To choose the id yourself, use `--id-column NAME`. An unknown name, or a column with blanks or under 99 percent distinct values, stops the run with a one-line error. After the run, check that the "Record id column" line near the top of `rules_report.md` names the column you expect.

## Options

- `--out DIR`: output folder, created if missing (required).
- `--seed N`: random seed, default 42.
- `--max-rows N`: analyse at most N rows of a larger file, default 20000.
- `--sample-mode groups|random`: above the cap, `groups` keeps rows that share a distinctive value together (so duplicate sets survive the cut); `random` draws rows uniformly.
- `--exclude-columns A,B`: leave these columns out even if Verity would use them.
- `--id-column NAME`: use this column as the record id (no blanks, at least 99% distinct values).
- `--keep-columns A,B`: use these columns even if a Verity rule would exclude them (not the record id column).
- `--mask-columns A,B`: replace these columns' values with `***` in the pairs files and in `rules.json`.
- `--no-pairs`: write no pairs files and leave `calibration_pairs` empty.
- `--debug`: also write `debug.json` (internal scores and a threshold, not part of the rules).

## Reading the report

Open `rules_report.md`. Start with these parts:

- **Columns, by weight.** Weight is how much the column should count (weights sum to 1). Reliability says how much more alike its values are across likely duplicates than across random pairs (0 is chance, 1 is always the same). "Low confidence" means too few anchor pairs, so treat that column's weight as less certain. The reason sentence says why.
- **Comparison method.** How to compare two values: `email`, `bigram_jaccard`, `text_bigram` and so on; the glossary in the report defines each.
- **Excluded columns.** Each has a reason and, if a Verity rule excluded it, the `--keep-columns` hint. Check that nothing you need was dropped.
- **Expected pair coverage.** The chance a typical pair has a value in both records of the kept columns, from 0 to 1. Below about 0.3, many true duplicates share too few columns to be decided; send those to review. That 0.3 comes from synthetic tests.
- **Notes.** Relaxed anchor rules, short calibration categories, sampling.

A weight describes how distinctive and consistent a column is in this data. Read weights together with the reasons.

## Label the pairs and pick a cutoff

1. Open `calibration_pairs.csv` and fill `your_label` (match or not a match) for each pair. Label at least 30 of each kind; add pairs from your own data if the file has fewer.
2. Do not open `calibration_pairs_key.csv` until you finish. It holds Verity's guesses and the pair sources, which are not the truth.
3. Check first whether your tool can use these weights. In your matching tool, score every labelled pair with the report's formula: sum of weight x similarity over columns present in both records, divided by the sum of those columns' weights. The similarity for each column is your tool's nearest comparator for the method named in the report (edit distance or Jaro-Winkler for text, exact for ids), so the cutoff belongs to that tool.
4. Sort by score. For each candidate cutoff count wrong calls (matches below it plus non-matches at or above it). Pick the cutoff with the fewest; on a tie, the middle of the tied range. Move it up if a false merge costs more than a missed match.
5. Check whether low-coverage pairs are decided worse than the rest; if so, route them to review.

## Handling the pairs files and rules.json

`calibration_pairs.csv`, `calibration_pairs_key.csv` and the pairs inside `rules.json` contain values from the source data. Handle them like the source data: same access, storage and retention. For sensitive columns (names, SSN, phone, email) use `--mask-columns`; masked pairs are harder to label, so mask only what you must. `--no-pairs` removes pairs entirely but then you have nothing to calibrate with. Check the output folder before sending it anywhere.

## Limits

- Tested on synthetic customer and product data, not on real customer data. See `docs/KNOWN_LIMITS.md`.
- Weights come from one run and move with the sample: on a 4,000-row file, 25% subsets moved weights by about 0.03 and shuffled low ranks.
- Files over `--max-rows` are cut, so duplicates split by the cut are not seen. Raise `--max-rows` if memory and time allow (100,000 rows took about 90 seconds).
- Verity does not match, clean or standardize records, and gives no cutoff.
- Candidate pairs need a shared value in a distinctive column. True duplicates that share none are not used to measure reliability.

## What to send back to the project owner

After each run send: the Verity version (`verity --version`), the command line used, `rules_report.md`, and a short note with row count, the column types you expected versus the report, any excluded column you disagree with, and how many pairs you labelled. Send `rules.json` and the pairs files only if the data owner agrees, or send masked versions. Do not send the source CSV.
