"""Text normalisation shared by both languages.

Currently handles full-width -> half-width folding, whitespace collapsing and
integer spell-out. It is intentionally lightweight; a fuller normaliser
(dates, currencies, units) is future work.
"""

from __future__ import annotations

import re

_HAN_RE = re.compile(r"[一-鿿]")

_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ["", "十", "百", "千"]

_EN_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def full_to_half(text: str) -> str:
    """Fold full-width ASCII / punctuation and the ideographic space to ASCII."""
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def number_to_chinese(n: int) -> str:
    """Spell a non-negative integer (< 10000) in Mandarin."""
    if n < 0:
        return "负" + number_to_chinese(-n)
    if n == 0:
        return _CN_DIGITS[0]
    digits = [int(c) for c in str(n)]
    length = len(digits)
    parts: list[str] = []
    zero_pending = False
    for i, d in enumerate(digits):
        pos = length - 1 - i
        if d == 0:
            zero_pending = True
        else:
            if zero_pending:
                parts.append(_CN_DIGITS[0])
                zero_pending = False
            parts.append(_CN_DIGITS[d] + _CN_UNITS[pos % 4])
    result = "".join(parts)
    if result.startswith("一十"):  # 十X reads without the leading 一
        result = result[1:]
    return result


def _below_hundred_en(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    tens, ones = divmod(n, 10)
    return _EN_TENS[tens] + ("" if ones == 0 else " " + _EN_ONES[ones])


def number_to_english(n: int) -> str:
    """Spell a non-negative integer (< 10000) in English."""
    if n < 0:
        return "negative " + number_to_english(-n)
    if n < 100:
        return _below_hundred_en(n)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        text = _EN_ONES[hundreds] + " hundred"
        return text if rest == 0 else text + " " + _below_hundred_en(rest)
    thousands, rest = divmod(n, 1000)
    text = _below_hundred_en(thousands) + " thousand"
    if rest == 0:
        return text
    return text + " " + (_below_hundred_en(rest) if rest < 100 else number_to_english(rest))


def normalize(text: str, language: str = "auto") -> str:
    """Fold width, collapse whitespace and spell out integer runs."""
    text = full_to_half(text)
    lang = language
    if lang == "auto":
        lang = "zh" if _HAN_RE.search(text) else "en"

    def _spell(match: re.Match) -> str:
        n = int(match.group())
        if lang == "zh":
            return number_to_chinese(n)
        return " " + number_to_english(n) + " "

    text = re.sub(r"\d+", _spell, text)
    return collapse_whitespace(text)
