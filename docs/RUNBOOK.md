# Verity runbook

Verity reads a CSV and learns which columns identify an entity and how much each should count. It writes rules (weights, comparison methods, reasons) and sample pairs to label. It does not match, merge or clean records, and it returns no match cutoff.

## When to use Verity, and when not

Use it when you have a table that probably holds duplicates and you need to know which columns to compare and how heavily, before you configure a matching tool. Do not use it to find or merge duplicates, to clean data, or to get a ready cutoff. It needs at least 2 rows; if the file has no unique id column, row numbers are used. With no or few shared values in distinctive columns it finds no candidate pairs and falls back to uniqueness-only weights (the run prints a warning).

Verity is not meant for transaction or event tables that repeat each customer many times. Reliability is only measured when some column has at least half of its values distinct. If duplicates repeat identical values, that holds up to about 2 rows per entity. If duplicates differ in spelling or format, it can hold with more. If no column has at least half of its values distinct, reliability is not measured and the report says so: a "Read this first" box opens the report, every reliability is 1.0, and the weights come from distinctiveness only.

## Install

Needs Python 3.10 or newer. Wheel location: <ADD PATH>

In a fresh virtual environment:

```
pip install verity_rules-0.1.8-py3-none-any.whl
verity --version
```

`pip install "verity_rules-0.1.8-py3-none-any.whl[debug]"` adds scikit-learn, used only for the debug threshold in `--debug`.

## Run

```
verity learn customers.csv --out out/customers
```

It writes five files to the folder: `rules.json`, `rules_report.md`, `columns.csv` (one row per column with every per-column field, to open in Excel), `calibration_pairs.csv`, `calibration_pairs_key.csv`. Same file and seed give the same output. If any of these files already exists in the folder, the run stops with a one-line error naming them; add `--overwrite` to replace them. If a file cannot be written (for example it is open in Excel), the run stops with a one-line error naming the path; close the file and run again.

Run time and memory: about 1 minute and 200 MB for 20,000 rows; about 90 seconds and 350 MB for 100,000 rows with `--max-rows 100000`.

### Check the file first

```
verity check customers.csv
```

Run this before `learn`. It writes nothing. It takes `--encoding` and `--delimiter` like `learn`. It prints the rows, the record id it would use (or "row numbers"), and for each column the share of non-blank values that are distinct and the share that are blank. It lists the columns that could produce candidate pairs (at least half of their values distinct, values shared by 2 to 10 rows), the number of candidate pairs, and one line on the expected outcome: reliability can be measured for every column; can be measured only partly (it lists which columns); or would not be measured, with the reason. When the outcome is "only partly" or "would not be measured", the next line says: "Verity may not suit this file. This usually means each entity appears in many rows, or duplicates differ in every column." If the most distinctive column repeats its values in groups with a median above 10 rows, it adds: "This looks like a table with many rows per entity, so Verity may not suit it." It uses the same functions as `learn`, so the candidate pair counts match. Counts cover the whole file; `learn` analyses at most `--max-rows` rows.

### File encoding and delimiter

Verity reads the file as UTF-8 (a byte order mark is dropped). If the bytes are not valid UTF-8 it reads the file as cp1252 and prints a one-line note. It never skips or replaces bytes it cannot decode; a file that fits neither stops the run with a one-line error. The delimiter is detected from the first 20 lines among comma, semicolon, tab and pipe. Both are recorded under "Run details" in `rules_report.md` and in `rules.json`. If only one column is read, the run stops with a one-line error: check the delimiter. Use `--encoding` and `--delimiter` to override detection.

### The record id column

A record id has no blank values and at least 99 percent distinct values. Verity picks one that qualifies, preferring names such as `id`, `record_id`, `uuid`, `key` or ending in `_id` or `_key`, and leaves it out of the weights. The pairs files show its values.

If no column qualifies, Verity does not pick one. It adds a row-number column named `_row` (1-based position in the file after the header; `_row_2` and so on if the file already has `_row`) and uses it as the record id. The run prints a warning, and `rules.json` notes and the report say: "No unique id column was found. Row numbers (_row) are used as record ids in the pairs files." Every other column, such as email, stays in the weights.

If Verity picks a column whose name does not read as an id (it is not `id`, `record_id`, `uuid` or `key`, and does not end in `_id` or `_key`), the run prints a warning, and `rules.json` notes and the report say: "Using <col> as the record id because it has no blanks and almost every value is distinct. If this is not an id, use --id-column or add row numbers." Nothing is printed when the name reads as an id or when you pass `--id-column`.

To choose the id yourself, use `--id-column NAME`. An unknown name, or a column with blanks or under 99 percent distinct values, stops the run with a one-line error. After the run, check that the "Record id column" line near the top of `rules_report.md` names the column you expect.

## Options

- `--out DIR`: output folder, created if missing (required).
- `--seed N`: random seed, default 42.
- `--max-rows N`: analyse at most N rows of a larger file, default 20000.
- `--sample-mode groups|random`: above the cap, `groups` keeps rows that share a distinctive value together (so duplicate sets survive the cut); `random` draws rows uniformly.
- `--exclude-columns A,B`: leave these columns out even if Verity would use them.
- `--encoding NAME`: text encoding of the CSV, such as `utf-8` or `cp1252` (default: UTF-8, then cp1252).
- `--delimiter CHAR`: `comma`, `semicolon`, `tab`, `pipe` or one character (default: detected from the first 20 lines).
- `--id-column NAME`: use this column as the record id (no blanks, at least 99% distinct values).
- `--keep-columns A,B`: use these columns even if a Verity rule would exclude them (not the record id column).
- `--mask-columns A,B`: replace these columns' values with `***` in the pairs files and in `rules.json`.
- `--overwrite`: replace output files that already exist in `--out` (default: refuse).
- `--no-pairs`: write no pairs files and leave `calibration_pairs` empty.
- `--debug`: also write `debug.json` (internal scores and a threshold, not part of the rules).

## Reading the report

Open `rules_report.md`. Start with these parts:

- **Summary paragraph and "Read this first" box.** The paragraph under the title gives rows analysed, columns kept, the three heaviest columns and whether reliability was measured. The box appears only when no candidate pairs were found; then reliability was not measured.
- **Columns, by weight.** The table shows Column, Weight, Reliability, Compared by and a one-line Reason. Weight is how much the column should count (weights sum to 1). Reliability says how much more alike its values are across likely duplicates than across random pairs (0 is chance, 1 is always the same). Directly under the table are the Caution about weights and a reminder to compare the weights with what you know about the data: a column that is generally a weak identifier can outrank a stronger one if its values are more distinct here. "Low confidence" means the relaxed anchor rule was used. "Not measured" means too few anchor pairs, so reliability is 1.0 and does not change the weight; treat that column's weight as less certain. The report defines each of these two markers only when at least one column carries it. The anchor counts are in the Details table below.
- **Comparison method.** How to compare two values: `email`, `bigram_jaccard`, `text_bigram` and so on; the glossary in the report defines each. If your tool does not have that comparator, use its closest one: edit distance or Jaro-Winkler for text, exact match for ids; for email, compare the part before the @ and the domain separately if you can. The weights stay valid. Set the cutoff using the comparator you actually use, because a cutoff only makes sense for the scoring it was chosen with.
- **Excluded columns.** Each has a reason and, if a Verity rule excluded it, the `--keep-columns` hint. Check that nothing you need was dropped.
- **Expected pair coverage.** The chance a typical pair has a value in both records of the kept columns, from 0 to 1. Below about 0.3, many true duplicates share too few columns to be decided; send those to review. That 0.3 comes from synthetic tests.
- **Notes.** Relaxed anchor rules, short calibration categories, sampling.

A weight describes how distinctive and consistent a column is in this data. Read weights together with the reasons.

## Label the pairs and pick a cutoff

If the run found no likely matches, the report's cutoff section is short: it says `calibration_pairs.csv` cannot be used to choose a cutoff and tells you to label pairs from your own data and follow steps 2 to 6.

1. Open `calibration_pairs.csv` and fill `your_label` (match or not a match) for each pair. Label at least 30 of each kind; add pairs from your own data if the file has fewer.
2. Do not open `calibration_pairs_key.csv` until you finish. It holds Verity's guesses and the pair sources, which are not the truth.
3. Check first whether your tool can use these weights. In your matching tool, score every labelled pair with the report's formula: sum of weight x similarity over columns present in both records, divided by the sum of those columns' weights. The similarity for each column is your tool's nearest comparator for the method named in the report (edit distance or Jaro-Winkler for text, exact for ids), so the cutoff belongs to that tool.
4. Sort by score. For each candidate cutoff count wrong calls (matches below it plus non-matches at or above it). Pick the cutoff with the fewest; on a tie, the middle of the tied range. Move it up if a false merge costs more than a missed match.
5. Check whether low-coverage pairs are decided worse than the rest; if so, route them to review.

## Opening the files in Excel

Excel drops leading zeros and rewrites long numbers (for example `00123` becomes `123`, and a 16-digit number becomes `1.23457E+15`) when it opens a CSV, and a file saved from Excel keeps those changes. So label inside the file, but score from the source records, looking each pair up by its record ids. Do not score from the values in the labelled file. Keep the exact CSV you ran, because `_row` numbers refer to the position of rows in that file; if you edit, sort or re-export the source, they no longer line up.

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
