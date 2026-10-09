"""Rerun into the same --out, --overwrite, and a failed write. All data is built here."""
import csv
from pathlib import Path

import pytest

from verity_rules.cli import OUTPUT_FILES, main

FIRSTS = ["Ann", "Bob", "Cara", "Dev", "Eli", "Fay", "Gus", "Hal"]


def _src(tmp_path):
    path = tmp_path / "in.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rec_id", "given", "family", "mail"])
        for i in range(60):
            p = i // 2
            first = FIRSTS[p % len(FIRSTS)]
            w.writerow([f"R{i:03d}", first + ("x" if i % 2 else ""), f"Name{p}",
                        f"{first}.name{p}@example.com".lower()])
    return path


def test_rerun_refuses_and_names_the_files(tmp_path, capsys):
    src, out = _src(tmp_path), tmp_path / "out"
    assert main(["learn", str(src), "--out", str(out)]) == 0
    before = {f: (out / f).read_bytes() for f in OUTPUT_FILES}
    capsys.readouterr()
    assert main(["learn", str(src), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("error: ")
    assert all(f in err for f in OUTPUT_FILES) and "--overwrite" in err
    assert {f: (out / f).read_bytes() for f in OUTPUT_FILES} == before


def test_rerun_names_only_the_files_that_exist(tmp_path, capsys):
    src, out = _src(tmp_path), tmp_path / "out"
    out.mkdir()
    (out / "rules.json").write_text("{}", encoding="utf-8")
    assert main(["learn", str(src), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "rules.json already exists" in err and "rules_report.md" not in err
    assert (out / "rules.json").read_text(encoding="utf-8") == "{}"


def test_overwrite_replaces_files(tmp_path):
    src, out = _src(tmp_path), tmp_path / "out"
    assert main(["learn", str(src), "--out", str(out)]) == 0
    first = (out / "rules.json").read_bytes()
    (out / "rules.json").write_text("{}", encoding="utf-8")
    assert main(["learn", str(src), "--out", str(out), "--overwrite"]) == 0
    assert (out / "rules.json").read_bytes() == first


@pytest.mark.parametrize("failing", ["rules.json", "rules_report.md", "calibration_pairs.csv"])
def test_write_failure_is_one_line_and_exit_1(tmp_path, capsys, monkeypatch, failing):
    src, out = _src(tmp_path), tmp_path / "out"
    real_write, real_open = Path.write_text, Path.open

    def deny(path):
        return PermissionError(13, "Permission denied", str(path))

    def write_text(self, *a, **k):
        if self.name == failing:
            raise deny(self)
        return real_write(self, *a, **k)

    def open_(self, *a, **k):
        if self.name == failing:
            raise deny(self)
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "open", open_)
    assert main(["learn", str(src), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("error: cannot write ")
    assert failing in err and "Excel" in err and "close it" in err
    assert "Traceback" not in err
