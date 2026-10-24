"""Unit tests for the gaps.py."""

import pytest

from src.actions_server.gaps import normalize, validate
from src.constants import Constants

GOOD_URL = "https://example.org/a"


def test_normalize_collapses_case_and_whitespace():
    assert normalize("  Early   SIGNS ") == "early signs"


def test_valid_input_passes():
    validate("Early signs of autism?", [GOOD_URL])


def test_empty_url_list_is_allowed():
    validate("Early signs of autism?", [])


def test_scheme_is_case_insensitive():
    validate("q", ["HTTPS://example.org/a"])


def test_limits_are_inclusive():
    validate("a" * Constants.MAX_QUESTION_LEN, [GOOD_URL] * Constants.MAX_URLS_PER_GAP)


@pytest.mark.parametrize("question", [None, 123, "", "   "])
def test_bad_question_rejected(question):
    with pytest.raises(ValueError):
        validate(question, [])


def test_question_too_long_rejected():
    with pytest.raises(ValueError):
        validate("a" * (Constants.MAX_QUESTION_LEN + 1), [])


@pytest.mark.parametrize("urls", [
    GOOD_URL,                                          # not a list
    [GOOD_URL] * (Constants.MAX_URLS_PER_GAP + 1),     # too many
    [123],                                             # not a string
    ["javascript:alert(1)"],
    ["ftp://example.org/x"],
    ["http://"],                                       # no host
    ["https://example.org/" + "a" * Constants.MAX_URL_LEN],  # too long
])
def test_bad_urls_rejected(urls):
    with pytest.raises(ValueError):
        validate("q", urls)