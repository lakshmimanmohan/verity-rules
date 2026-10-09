"""Encoding and delimiter handling in the CLI reader. All data is built here."""
import csv
import io
import json

import pytest

from verity_rules.cli import main, read_table, sniff_delimiter

NAMES = [("José", "Müller", "Zürich"), ("Renée", "Dupont", "Orléans"), ("Zoë", "Smith", "Köln"),
         ("Ana", "García", "Málaga"), ("Bob", "Jones", "Leeds"), ("Eve", "Brown", "York")]


def _text(delimiter):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    w.writerow(["cust_id", "first_name", "last_name", "email", "city"])
    for i in range(30):
        f, l, c = NAMES[i % len(NAMES)]
        w.writerow([f"C{i:04d}", f, l, f"{f.lower()}{i}@x.com", c])
    return buf.getvalue()


def _file(tmp_path, name, delimiter=",", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(_text(delimiter).encode(encoding))
    return path


def _run(tmp_path, src, *extra):
    out = tmp_path / "out"
    code = main(["learn", str(src), "--out", str(out), *extra])
    return code, out


def _run_info(out):
    return json.loads((out / "rules.json").read_text(encoding="utf-8"))["run"]


@pytest.mark.parametrize("name,delimiter,encoding,enc_used,delim_used", [
    ("bom.csv", ",", "utf-8-sig", "utf-8-sig", "comma"),
    ("plain.csv", ",", "utf-8", "utf-8", "comma"),
    ("cp1252.csv", ",", "cp1252", "cp1252", "comma"),
    ("semi.csv", ";", "utf-8", "utf-8", "semicolon"),
    ("tab.csv", "\t", "utf-8", "utf-8", "tab"),
    ("pipe.csv", "|", "utf-8", "utf-8", "pipe"),
])
def test_detected_and_recorded(tmp_path, name, delimiter, encoding, enc_used, delim_used):
    src = _file(tmp_path, name, delimiter, encoding)
    records, info = read_table(src)
    assert info == {"encoding": enc_used, "delimiter": delim_used}
    assert list(records[0]) == ["cust_id", "first_name", "last_name", "email", "city"]
    assert records[0]["first_name"] == "José" and records[3]["last_name"] == "García"
    code, out = _run(tmp_path, src)
    assert code == 0
    run = _run_info(out)
    assert (run["encoding"], run["delimiter"]) == (enc_used, delim_used)
    report = (out / "rules_report.md").read_text(encoding="utf-8")
    assert f"- File encoding: {enc_used}" in report and f"- Delimiter: {delim_used}" in report


def test_cp1252_fallback_says_so(tmp_path, capsys):
    code, _ = _run(tmp_path, _file(tmp_path, "c.csv", encoding="cp1252"))
    assert code == 0 and "read as cp1252" in capsys.readouterr().err


def test_utf8_does_not_print_fallback_note(tmp_path, capsys):
    _run(tmp_path, _file(tmp_path, "u.csv", ";"))
    assert "cp1252" not in capsys.readouterr().err


def test_bom_is_not_in_the_first_column_name(tmp_path):
    records, _ = read_table(_file(tmp_path, "b.csv", encoding="utf-8-sig"))
    assert "cust_id" in records[0]


def test_bytes_undefined_in_cp1252_stop_the_run(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a,b\n\x81,1\n\xe9,2\n")
    assert main(["learn", str(path), "--out", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "neither UTF-8 nor cp1252" in err and "--encoding" in err


def test_encoding_override_decodes_strictly(tmp_path, capsys):
    src = _file(tmp_path, "c.csv", encoding="cp1252")
    assert main(["learn", str(src), "--out", str(tmp_path / "o1"), "--encoding", "utf-8"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "not valid utf-8" in err
    assert main(["learn", str(src), "--out", str(tmp_path / "o2"), "--encoding", "cp1252"]) == 0
    assert _run_info(tmp_path / "o2")["encoding"] == "cp1252"


def test_unknown_encoding_and_delimiter(tmp_path, capsys):
    src = _file(tmp_path, "s.csv")
    assert main(["learn", str(src), "--out", str(tmp_path / "o"), "--encoding", "nope"]) == 1
    assert "unknown encoding" in capsys.readouterr().err
    assert main(["learn", str(src), "--out", str(tmp_path / "o"), "--delimiter", "xx"]) == 1
    assert "--delimiter" in capsys.readouterr().err


def test_delimiter_override_wins_over_sniffing(tmp_path):
    src = _file(tmp_path, "s.csv", ";")
    code, out = _run(tmp_path, src, "--delimiter", "semicolon")
    assert code == 0 and _run_info(out)["delimiter"] == "semicolon"


def test_one_column_result_stops_with_delimiter_hint(tmp_path, capsys):
    src = _file(tmp_path, "s.csv", ";")
    assert main(["learn", str(src), "--out", str(tmp_path / "o"), "--delimiter", "comma"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "only one column" in err and "check the delimiter" in err
    assert not (tmp_path / "o").exists()


def test_sniff_ignores_a_delimiter_inside_a_few_values():
    text = "a,b,c\n" + "".join(f"x{i},1 Main St; Apt {i},z\n" for i in range(5))
    assert sniff_delimiter(text) == ","


def test_sniff_falls_back_to_comma():
    assert sniff_delimiter("onlyone\nvalue\n") == ","
