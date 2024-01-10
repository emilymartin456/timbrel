"""Language segmentation for code-switched (mixed zh/en) text.

Splits a string into maximal runs of a single class so the bilingual front-end
can route each run to the right g2p. Classes are ``zh`` (CJK), ``en`` (Latin
letters / apostrophe) and ``other`` (whitespace, punctuation, digits).
"""

from __future__ import annotations

import re

_ZH_RE = re.compile(r"[一-鿿㐀-䶿]")


def char_language(ch: str) -> str:
    if _ZH_RE.match(ch):
        return "zh"
    if ch.isascii() and (ch.isalpha() or ch == "'"):
        return "en"
    return "other"


def segment(text: str) -> list[tuple[str, str]]:
    """Return ``[(language, chunk), ...]`` runs in order."""
    runs: list[tuple[str, str]] = []
    current: str | None = None
    buffer = ""
    for ch in text:
        lang = char_language(ch)
        if lang == current:
            buffer += ch
        else:
            if buffer:
                runs.append((current, buffer))  # type: ignore[arg-type]
            current = lang
            buffer = ch
    if buffer:
        runs.append((current, buffer))  # type: ignore[arg-type]
    return runs
