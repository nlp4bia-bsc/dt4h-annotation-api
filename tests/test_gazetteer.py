"""Gazetteer loading — the contract every index row position depends on."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.src.nel.gazetteer import fingerprint, load_gazetteer


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "gaz.tsv"
    path.write_text(body, encoding="utf-8")
    return path


def test_canonical_columns_and_positional_index(gaz_path: Path):
    rows = load_gazetteer(gaz_path)
    assert list(rows.columns) == ["term", "code"]
    assert list(rows.index) == list(range(len(rows)))
    assert rows["term"].tolist()[0] == "meningitis bacteriana"


def test_codes_stay_strings_when_the_column_looks_numeric(tmp_path: Path):
    """A blank code makes pandas infer float64; str(840539006.0) is not a code.

    This is a regression guard: every SNOMED code would otherwise gain a
    ``.0`` suffix as soon as one row had a missing code.
    """
    path = write(tmp_path, "term\tcode\ncovid\t840539006\northan\t\n")
    rows = load_gazetteer(path)
    assert rows["code"].tolist() == ["840539006"]


def test_duplicate_terms_keep_the_first_code(tmp_path: Path):
    path = write(tmp_path, "term\tcode\ncovid\t111\ncovid\t222\nflu\t333\n")
    rows = load_gazetteer(path)
    assert rows["term"].tolist() == ["covid", "flu"]
    assert rows["code"].tolist() == ["111", "333"]


def test_whitespace_is_stripped_before_deduplication(tmp_path: Path):
    path = write(tmp_path, "term\tcode\nmeningitis\t111\n  meningitis  \t222\n")
    rows = load_gazetteer(path)
    assert len(rows) == 1


def test_rows_with_an_empty_term_or_code_are_dropped(tmp_path: Path):
    path = write(tmp_path, "term\tcode\ncovid\t111\n\t222\northan\t\n")
    rows = load_gazetteer(path)
    assert rows["term"].tolist() == ["covid"]


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_gazetteer(tmp_path / "absent.tsv")


def test_missing_required_column_names_it(tmp_path: Path):
    path = write(tmp_path, "term\tconcept\ncovid\t111\n")
    with pytest.raises(ValueError, match="code"):
        load_gazetteer(path)


def test_gazetteer_with_no_usable_rows_raises(tmp_path: Path):
    path = write(tmp_path, "term\tcode\n\t\n")
    with pytest.raises(ValueError, match="no usable"):
        load_gazetteer(path)


def test_fingerprint_is_stable_and_content_sensitive(gaz_path: Path):
    before = fingerprint(gaz_path)
    assert before == fingerprint(gaz_path)
    gaz_path.write_text(gaz_path.read_text(encoding="utf-8") + "gripe\t6142004\n", encoding="utf-8")
    assert fingerprint(gaz_path) != before


def test_fingerprint_detects_reordering_without_row_changes(tmp_path: Path):
    """Row *order* is the contract, not just the row set."""
    one = write(tmp_path, "term\tcode\na\t1\nb\t2\n")
    first = fingerprint(one)
    one.write_text("term\tcode\nb\t2\na\t1\n", encoding="utf-8")
    assert fingerprint(one) != first
