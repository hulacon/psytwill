"""The ext-<corpus>-<native> fit-corpus namespace."""

import pytest

from psytwill.exceptions import CorpusError
from psytwill.fitcorpus import (
    CORPORA,
    _validate_registry,
    CorpusSpec,
    ext_id,
    is_external,
    parse_ext_id,
)


class TestExtId:
    def test_string_native(self):
        assert ext_id("movie10", "bourne-seg012") == "ext-movie10-bourne-seg012"

    def test_int_native_zero_padded_to_corpus_width(self):
        assert ext_id("nsd", 21384) == "ext-nsd-021384"
        assert ext_id("nsd", 1) == "ext-nsd-000001"

    def test_unknown_corpus_names_registry(self):
        with pytest.raises(CorpusError, match="unknown fit corpus"):
            ext_id("imagenet", "n01440764")

    def test_bad_native_charset_refused(self):
        with pytest.raises(CorpusError, match="normalize upstream"):
            ext_id("librispeech", "Chapter_19")

    def test_negative_int_refused(self):
        with pytest.raises(CorpusError, match="non-negative"):
            ext_id("nsd", -1)


class TestParseExtId:
    def test_round_trip_every_registered_corpus(self):
        for key in CORPORA:
            sid = ext_id(key, "unit-0042")
            assert parse_ext_id(sid) == (key, "unit-0042")

    def test_longest_match_wins_for_hyphenated_keys(self):
        # "peoples-speech" must not split at its own internal hyphen
        corpus, native = parse_ext_id("ext-peoples-speech-chunk0042")
        assert corpus == "peoples-speech"
        assert native == "chunk0042"

    def test_hyphens_in_native_survive(self):
        assert parse_ext_id("ext-movie10-bourne-seg012") == (
            "movie10", "bourne-seg012"
        )

    def test_non_external_id_refused(self):
        with pytest.raises(CorpusError, match="not an external"):
            parse_ext_id("shared0001_nsd02951")

    def test_unregistered_corpus_refused(self):
        with pytest.raises(CorpusError, match="names no registered corpus"):
            parse_ext_id("ext-imagenet-n01440764")

    def test_empty_native_refused(self):
        with pytest.raises(CorpusError, match="empty native id"):
            parse_ext_id("ext-nsd")


class TestIsExternal:
    def test_partition(self):
        assert is_external("ext-nsd-000001")
        assert not is_external("shared0001_nsd02951")
        assert not is_external("adventure-time")


class TestRegistryInvariants:
    def test_shipped_registry_valid(self):
        _validate_registry(CORPORA)

    def test_prefix_ambiguous_keys_refused(self):
        bad = {
            "twp": CorpusSpec("twp", tier="local", serves="x"),
            "twp-unpresented": CorpusSpec(
                "twp-unpresented", tier="local", serves="x"
            ),
        }
        with pytest.raises(CorpusError, match="parse-ambiguous"):
            _validate_registry(bad)

    def test_bad_key_charset_refused(self):
        bad = {"NSD": CorpusSpec("NSD", tier="local", serves="x")}
        with pytest.raises(CorpusError, match="lowercase"):
            _validate_registry(bad)

    def test_ledger_tiers_only(self):
        assert {s.tier for s in CORPORA.values()} <= {
            "local", "a", "b", "validation"
        }
