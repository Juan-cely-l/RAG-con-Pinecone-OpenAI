"""Tests for validation.validate_question."""

import pytest

from errors import InvalidInputError
from validation import validate_question


def test_valid_question_is_returned_stripped():
    assert validate_question("  How are rounds scored?  ", 500) == "How are rounds scored?"


def test_whitespace_is_collapsed_and_control_chars_removed():
    result = validate_question("What\tis a\n\n10-8\x00 round?\x07", 500)
    assert result == "What is a 10-8 round?"


def test_spanish_accents_and_inverted_question_mark_are_preserved():
    question = "¿Cuántos asaltos tiene una pelea de campeonato?"
    assert validate_question(question, 500) == question


@pytest.mark.parametrize("bad", [None, "", "   ", "\n\t  "])
def test_empty_or_missing_question_is_rejected(bad):
    with pytest.raises(InvalidInputError):
        validate_question(bad, 500)


def test_question_longer_than_limit_is_rejected():
    with pytest.raises(InvalidInputError):
        validate_question("a" * 501, 500)


def test_question_at_exact_limit_is_accepted():
    assert validate_question("a" * 500, 500) == "a" * 500


@pytest.mark.parametrize("bad", ["???", "!!! ...", "¿?"])
def test_question_without_alphanumerics_is_rejected(bad):
    with pytest.raises(InvalidInputError):
        validate_question(bad, 500)
