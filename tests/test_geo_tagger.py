"""
Geo tagger tests.

Pure string matching against a fixed gazetteer — no API calls, no database,
so these are cheap to run. The important cases are the negative ones: a wrong
country tag is worse than a missing one, because the globe reads as factual.
"""

import json
from unittest.mock import MagicMock

import pytest

from pipeline.geo_tagger import (
    GAZETTEER,
    GeoTagger,
    MAX_COUNTRIES,
    country_name,
    extract_countries,
)


class TestExtractCountries:
    def test_explicit_country_name(self):
        assert extract_countries("Germany raises defence spending") == ["DE"]

    def test_capital_city_implies_country(self):
        assert "CN" in extract_countries("Beijing tightens export controls")

    def test_central_bank_implies_country(self):
        assert extract_countries("Bank of Japan holds rates") == ["JP"]

    def test_uppercase_us_is_matched(self):
        assert "US" in extract_countries("US inflation cools in June")

    def test_capitalised_fed_is_matched(self):
        assert "US" in extract_countries("Fed holds rates steady")

    def test_lowercase_fed_is_not_a_country(self):
        """'fed' the verb must not be read as the Federal Reserve."""
        assert extract_countries("He fed the dog") == []

    def test_lowercase_us_pronoun_is_not_a_country(self):
        assert extract_countries("The results told us very little") == []

    def test_generic_business_copy_yields_nothing(self):
        """The Lockheed case: 'debut' with no geography must not tag a country."""
        assert extract_countries("Lockheed Martin fighter jet makes its debut") == []

    def test_multiple_countries_are_ranked_by_mentions(self):
        codes = extract_countries(
            "Beijing and Shanghai respond as Washington weighs new tariffs on China"
        )
        assert codes[0] == "CN", f"China is the subject but ranked {codes}"
        assert "US" in codes

    def test_respects_the_limit(self):
        codes = extract_countries(
            "Germany, France, Italy, Spain and Poland agree a joint position",
            limit=3,
        )
        assert len(codes) <= 3

    def test_default_limit_is_capped(self):
        codes = extract_countries(
            "Germany France Italy Spain Poland Greece Portugal Ireland"
        )
        assert len(codes) <= MAX_COUNTRIES

    def test_handles_empty_and_none_input(self):
        assert extract_countries(None, "", None) == []

    def test_south_korea_not_shadowed_by_shorter_alias(self):
        assert "KR" in extract_countries("South Korea exports rebound")


class TestCountryName:
    def test_known_code(self):
        assert country_name("KR") == "South Korea"

    def test_is_case_insensitive(self):
        assert country_name("kr") == "South Korea"

    def test_supranational_extra(self):
        assert country_name("EU") == "European Union"

    def test_unknown_code_falls_back_to_itself(self):
        assert country_name("ZZ") == "ZZ"


class TestGazetteerIntegrity:
    def test_codes_are_two_letter_uppercase(self):
        for code in GAZETTEER:
            assert len(code) == 2 and code.isupper(), f"bad code: {code}"

    def test_aliases_are_lowercase(self):
        """Aliases are matched case-insensitively; storing them lowercase keeps
        the table honest about that."""
        for code, (_, aliases) in GAZETTEER.items():
            for alias in aliases:
                assert alias == alias.lower(), f"{code} alias not lowercase: {alias}"

    def test_no_duplicate_aliases_across_countries(self):
        seen: dict[str, str] = {}
        for code, (_, aliases) in GAZETTEER.items():
            for alias in aliases:
                assert alias not in seen, (
                    f"alias {alias!r} claimed by both {seen.get(alias)} and {code}"
                )
                seen[alias] = code


class TestBackfill:
    def _db_with_rows(self, rows):
        db = MagicMock()
        conn = MagicMock()
        db.connection.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = rows
        return db, conn

    def test_writes_codes_for_matched_articles(self):
        db, conn = self._db_with_rows([
            {
                "id": "a1",
                "headline": "Bank of Japan holds rates",
                "summary": "",
                "classification_summary": None,
            }
        ])
        assert GeoTagger(db).backfill(limit=10) == 1

        updates = [c for c in conn.execute.call_args_list if "UPDATE" in c.args[0]]
        assert updates, "no UPDATE issued"
        assert json.loads(updates[0].args[1][0]) == ["JP"]

    def test_marks_unmatched_articles_so_the_backlog_converges(self):
        """An article with no geography is still written, as an empty list."""
        db, conn = self._db_with_rows([
            {
                "id": "a2",
                "headline": "Quarterly results beat expectations",
                "summary": "",
                "classification_summary": None,
            }
        ])
        assert GeoTagger(db).backfill(limit=10) == 0

        updates = [c for c in conn.execute.call_args_list if "UPDATE" in c.args[0]]
        assert updates, "unmatched article was not written back — backlog would stall"
        assert json.loads(updates[0].args[1][0]) == []

    def test_empty_backlog_is_a_no_op(self):
        db, conn = self._db_with_rows([])
        assert GeoTagger(db).backfill(limit=10) == 0
        assert not [c for c in conn.execute.call_args_list if "UPDATE" in c.args[0]]
