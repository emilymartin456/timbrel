"""English grapheme-to-phoneme.

Known words are looked up in :data:`timbrel.frontend.en_lexicon.LEXICON`;
unknown words fall back to a crude letter-to-ARPAbet rule engine. The fallback
is deliberately simple — plugging in CMUdict / a neural g2p is on the roadmap.
"""

from __future__ import annotations

import re

from timbrel.frontend.en_lexicon import LEXICON
from timbrel.frontend.phonemes import SIL, SP

_TOKEN_RE = re.compile(r"[a-zA-Z']+|[.,!?;:]")
_STOP_PUNCT = set(".!?")
_PAUSE_PUNCT = set(",;:")

_DIGRAPHS: dict[str, list[str]] = {
    "ch": ["CH"],
    "sh": ["SH"],
    "th": ["TH"],
    "ph": ["F"],
    "ck": ["K"],
    "ng": ["NG"],
    "qu": ["K", "W"],
    "wh": ["W"],
    "gh": [],
    "ee": ["IY"],
    "oo": ["UW"],
    "ou": ["AW"],
    "ow": ["AW"],
    "ai": ["EY"],
    "ay": ["EY"],
    "oa": ["OW"],
    "ea": ["IY"],
    "ie": ["AY"],
    "oi": ["OY"],
    "oy": ["OY"],
    "au": ["AO"],
    "aw": ["AO"],
}
_SINGLES: dict[str, list[str]] = {
    "a": ["AE"],
    "e": ["EH"],
    "i": ["IH"],
    "o": ["AA"],
    "u": ["AH"],
    "b": ["B"],
    "c": ["K"],
    "d": ["D"],
    "f": ["F"],
    "g": ["G"],
    "h": ["HH"],
    "j": ["JH"],
    "k": ["K"],
    "l": ["L"],
    "m": ["M"],
    "n": ["N"],
    "p": ["P"],
    "q": ["K"],
    "r": ["R"],
    "s": ["S"],
    "t": ["T"],
    "v": ["V"],
    "w": ["W"],
    "x": ["K", "S"],
    "z": ["Z"],
}


def _rule_g2p(word: str) -> list[str]:
    """Approximate ARPAbet for an out-of-vocabulary word."""
    phones: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        pair = word[i : i + 2]
        if pair in _DIGRAPHS:
            phones.extend(_DIGRAPHS[pair])
            i += 2
            continue
        ch = word[i]
        if ch == "y":
            phones.append("Y" if i == 0 else "IY")
        elif ch in _SINGLES:
            phones.extend(_SINGLES[ch])
        # apostrophes and stray characters are skipped
        i += 1
    return phones


class EnglishG2P:
    """Convert English text into the bilingual phoneme alphabet."""

    def __init__(self, insert_pause: bool = True) -> None:
        self.insert_pause = insert_pause

    def to_phonemes(self, text: str) -> list[str]:
        phones: list[str] = []
        for token in _TOKEN_RE.findall(text):
            if token in _STOP_PUNCT:
                if self.insert_pause:
                    phones.append(SIL)
            elif token in _PAUSE_PUNCT:
                if self.insert_pause:
                    phones.append(SP)
            else:
                word = token.lower().strip("'")
                phones.extend(LEXICON.get(word) or _rule_g2p(word))
        return phones
