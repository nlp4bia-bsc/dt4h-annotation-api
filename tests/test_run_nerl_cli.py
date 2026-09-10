"""``run_nerl.py`` argument handling for the NEL method selection.

Only the parsing and the resource-check gate are exercised here — running the
script needs real NER checkpoints the offline suite will not download.  Both
are worth pinning anyway: ``--nel`` has to keep meaning what it meant before
methods were selectable, and a lexical-only run must not be blocked on a vector
DB it will never open.
"""

from __future__ import annotations

import pytest

import run_nerl
from run_nerl import DEFAULT_NEL_METHODS, NEL_METHOD_FLAGS, _resolve_nel_methods


# -- method selection ------------------------------------------------------


def test_absent_flag_means_ner_only():
    assert _resolve_nel_methods(None) is None


def test_bare_flag_keeps_meaning_dense():
    assert _resolve_nel_methods([]) == list(DEFAULT_NEL_METHODS)


def test_methods_are_deduplicated_and_put_in_pipeline_order():
    """CLI order is cosmetic: the linker always builds generators in one order.

    Reporting the methods in the order they were typed would imply the score
    tie-break follows the command line, which it does not.
    """
    assert _resolve_nel_methods(["bm25", "exact", "bm25"]) == ["exact", "bm25"]


def test_every_method_maps_to_a_pipeline_keyword():
    from app.src.pipelines import BiencoderPipeline

    parameters = BiencoderPipeline.__init__.__code__.co_varnames
    for flag in NEL_METHOD_FLAGS.values():
        assert flag in parameters, f"BiencoderPipeline has no '{flag}' parameter"


# -- argparse wiring -------------------------------------------------------


def _parse(argv: list[str], monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_nerl.py", *argv])
    return run_nerl._parse_args()


def test_parser_accepts_a_bare_flag_and_a_method_list(monkeypatch):
    assert _parse([], monkeypatch).nel is None
    assert _parse(["--nel"], monkeypatch).nel == []
    assert _parse(["--nel", "exact", "tfidf"], monkeypatch).nel == ["exact", "tfidf"]


def test_parser_still_reads_the_options_that_follow_the_method_list(monkeypatch):
    args = _parse(["--nel", "exact", "-l", "es", "-e", "disease"], monkeypatch)
    assert args.nel == ["exact"]
    assert args.langs == ["es"]
    assert args.entities == ["disease"]


def test_parser_rejects_an_unknown_method(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(["--nel", "levenshtein"], monkeypatch)


# -- resource gate ---------------------------------------------------------


class _Resolver:
    """Records what the check asked for; the gazetteer is the only thing present."""

    def __init__(self):
        self.asked = []

    def get_nel_path(self, lang):
        self.asked.append("nel")
        return None, "some/repo"  # not downloaded

    def get_gaz_path(self, lang, entity):
        self.asked.append("gaz")
        return f"/gaz/{lang}/{entity}.tsv"

    def get_vector_db_path(self, lang, entity):
        self.asked.append("vdb")
        return None, False  # not built


def test_dense_requires_the_encoder_and_a_built_index():
    resolver = _Resolver()
    assert run_nerl._check_nel_registry(resolver, "es", ["disease"], ["dense"]) is False
    assert "nel" in resolver.asked and "vdb" in resolver.asked


def test_lexical_only_passes_without_encoder_or_index():
    resolver = _Resolver()
    assert run_nerl._check_nel_registry(resolver, "es", ["disease"], ["exact", "bm25"]) is True
    assert resolver.asked == ["gaz"], "a lexical run must not demand dense resources"
