import pytest

from query_processing.preprocess import (
    classify_question,
    clean,
    detect_safety_flags,
    extract_keywords,
    normalize,
    preprocess,
)


def test_clean_collapses_whitespace_and_control_chars():
    assert clean("  what\tis\n\n this\x00 ?  ") == "what is this ?"


def test_clean_normalizes_unicode_and_smart_quotes():
    assert clean("what’s Ａ") == "what's A"


def test_normalize_lowercases_and_expands_contractions():
    assert normalize("What's the cause? It's unclear, isn't it") == "what is the cause? it is unclear, is not it"


def test_extract_keywords_removes_stopwords_and_duplicates():
    tokens = ["what", "is", "the", "x-ray", "result", "of", "the", "x-ray", "a"]
    assert extract_keywords(tokens) == ["x-ray", "result"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("what is term x", "definition"),
        ("what are the symptoms of term x", "symptoms"),
        ("what causes term x", "causes"),
        ("how is term x treated", "treatment"),
        ("how can term x be prevented", "prevention"),
        ("how is term x diagnosed", "diagnosis"),
        ("difference between term a and term b", "comparison"),
        ("tell me about term x", "general"),
    ],
)
def test_classify_question(text, expected):
    assert classify_question(text) == expected


def test_safety_flags():
    assert detect_safety_flags("i have chest pain right now") == ["possible_emergency"]
    assert detect_safety_flags("should i take two tablets") == ["personal_medical_advice"]
    assert detect_safety_flags("what is term x") == []


def test_preprocess_end_to_end():
    q = preprocess("  What's   the TREATMENT for Term-X? ")
    assert q.clean_text == "What's the TREATMENT for Term-X?"
    assert q.normalized_text == "what is the treatment for term-x?"
    assert q.keywords == ["treatment", "term-x"]
    assert q.question_type == "treatment"
    assert q.safety_flags == []
