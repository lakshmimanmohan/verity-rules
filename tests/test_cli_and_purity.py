"""Rule 2: only verity_rules/cli.py may touch files. Plus a CLI smoke test."""
import ast
import csv
import json
import random
from pathlib import Path

from verity_rules.cli import main

ROOT = Path(__file__).resolve().parents[1]
FILE_CALLS = {"open", "read_csv", "to_csv", "read_text", "write_text", "read_bytes",
              "write_bytes", "read_json", "to_json", "read_excel", "to_parquet"}
NETWORK_MODULES = {"requests", "urllib", "socket", "http", "httpx"}


def test_only_cli_touches_files_or_network():
    hits = []
    for path in sorted((ROOT / "verity_rules").rglob("*.py")):
        if path.name == "cli.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if name in FILE_CALLS:
                    hits.append(f"{path.name}:{node.lineno}: {name}()")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(m.split(".")[0] in NETWORK_MODULES for m in mods):
                    hits.append(f"{path.name}:{node.lineno}: network import")
    assert not hits, "file/network access outside cli.py:\n" + "\n".join(hits)


def _write_csv(path: Path, n: int = 80) -> None:
    rng = random.Random(1)
    firsts = ["Ann", "Bob", "Cara", "Dev", "Eli", "Fay", "Gus", "Hal"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rec_key", "given", "family", "mail"])
        for i in range(n):
            f, l = rng.choice(firsts), f"Name{rng.randrange(40)}"
            w.writerow([f"R{i:04d}", f, l, f"{f}.{l}@example.com".lower()])


def test_cli_learn_writes_three_files(tmp_path):
    src = tmp_path / "in.csv"
    _write_csv(src)
    out = tmp_path / "nested" / "out"
    assert main(["learn", str(src), "--out", str(out), "--debug"]) == 0
    for name in ("rules.json", "rules_report.md", "calibration_pairs.csv", "debug.json"):
        assert (out / name).is_file()
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    assert "threshold" not in rules and "similarity_stats" not in rules
    header = next(csv.reader((out / "calibration_pairs.csv").open(encoding="utf-8")))
    assert header[:4] == ["pair_id", "suggested_label", "record_id_a", "record_id_b"]
    assert header[-1] == "your_label"


def test_cli_max_rows_notes_original_count(tmp_path):
    src = tmp_path / "in.csv"
    _write_csv(src, 80)
    assert main(["learn", str(src), "--out", str(tmp_path / "o"), "--max-rows", "50"]) == 0
    rules = json.loads((tmp_path / "o" / "rules.json").read_text(encoding="utf-8"))
    assert rules["n_records"] == 50
    assert any("80" in n and "50" in n for n in rules["notes"])


def test_cli_friendly_errors(tmp_path, capsys):
    assert main(["learn", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "o")]) == 1
    assert "file not found" in capsys.readouterr().err
    one = tmp_path / "one.csv"
    one.write_text("a,b\n1,2\n", encoding="utf-8")
    assert main(["learn", str(one), "--out", str(tmp_path / "o")]) == 1
    assert "at least 2" in capsys.readouterr().err
