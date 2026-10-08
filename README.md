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
| `--debug` | off | Also write `debug.json` (internal scores and threshold; not part of the public rules) |

## Output

| File | Contents |
|---|---|
| `rules.json` | Learned rules: per-column weight, reliability, comparison method and reason, plus separability and calibration pairs |
| `rules_report.md` | Plain-language report of the rules, for people new to Verity |
| `calibration_pairs.csv` | Sample pairs to label by hand. Fill in `your_label` |

Set your own match cutoff in your own tool by scoring `calibration_pairs.csv` there. Verity does not return one.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```
