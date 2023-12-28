"""Mandarin grapheme-to-phoneme.

Chinese text is converted to tone-3 pinyin with :mod:`pypinyin`, then each
syllable is split into an *initial* (声母) and a tone-tagged *final* (韵母), e.g.
``中`` -> ``zhong1`` -> ``["zh", "ong1"]``.
"""

from __future__ import annotations

import re

from pypinyin import Style, pinyin

from timbrel.frontend.phonemes import MANDARIN_INITIALS, SIL, SP

# match multi-letter initials (zh/ch/sh) before single letters
_INITIALS_BY_LEN = sorted(MANDARIN_INITIALS, key=len, reverse=True)
_SYLLABLE_RE = re.compile(r"^([a-z]+)([1-5])$")

_PAUSE_PUNCT = set("，、,;:")
_STOP_PUNCT = set("。！？!?.…")


def _split_syllable(syllable: str) -> list[str]:
    """Split a tone-3 pinyin syllable into ``[initial, final+tone]`` phones."""
    match = _SYLLABLE_RE.match(syllable)
    if match is None:
        return []
    body, tone = match.group(1), match.group(2)
    initial = ""
    for candidate in _INITIALS_BY_LEN:
        if body.startswith(candidate):
            initial = candidate
            break
    final = body[len(initial) :]
    phones: list[str] = []
    if initial:
        phones.append(initial)
    if final:
        phones.append(final + tone)
    return phones


class ChineseG2P:
    """Convert Mandarin text into the bilingual phoneme alphabet."""

    def __init__(self, insert_pause: bool = True) -> None:
        self.insert_pause = insert_pause

    def _syllables(self, text: str) -> list[str]:
        pys = pinyin(
            text,
            style=Style.TONE3,
            strict=False,
            neutral_tone_with_five=True,
            errors=lambda chars: list(chars),
        )
        return [item[0] for item in pys]

    def to_phonemes(self, text: str) -> list[str]:
        phones: list[str] = []
        # TODO: apply third-tone sandhi before splitting syllables
        for token in self._syllables(text):
            split = _split_syllable(token)
            if split:
                phones.extend(split)
            elif self.insert_pause:
                for ch in token:
                    if ch in _STOP_PUNCT:
                        phones.append(SIL)
                    elif ch in _PAUSE_PUNCT:
                        phones.append(SP)
        return phones
