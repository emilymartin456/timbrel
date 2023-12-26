from timbrel.frontend.normalize import (
    collapse_whitespace,
    full_to_half,
    normalize,
    number_to_chinese,
    number_to_english,
)


def test_full_to_half():
    assert full_to_half("ＡＢＣ１２３") == "ABC123"


def test_collapse_whitespace():
    assert collapse_whitespace("a   b\t c\n") == "a b c"


def test_number_to_chinese():
    assert number_to_chinese(0) == "零"
    assert number_to_chinese(10) == "十"
    assert number_to_chinese(105) == "一百零五"
    assert number_to_chinese(2026) == "二千零二十六"


def test_number_to_english():
    assert number_to_english(7) == "seven"
    assert number_to_english(21) == "twenty one"
    assert number_to_english(305) == "three hundred five"


def test_normalize_numbers_by_language():
    assert normalize("我有3个苹果", "zh") == "我有三个苹果"
    assert "twenty one" in normalize("room 21", "en")
