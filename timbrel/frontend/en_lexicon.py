"""A small built-in English pronunciation lexicon (ARPAbet, no stress marks).

This covers the words used in examples/tests and the most frequent function
words. Out-of-vocabulary words fall back to the rule-based g2p in
:mod:`timbrel.frontend.english`.
"""

from __future__ import annotations

LEXICON: dict[str, list[str]] = {
    "a": ["AH"],
    "and": ["AH", "N", "D"],
    "are": ["AA", "R"],
    "audio": ["AO", "D", "IY", "OW"],
    "chinese": ["CH", "AY", "N", "IY", "Z"],
    "clone": ["K", "L", "OW", "N"],
    "english": ["IH", "NG", "G", "L", "IH", "SH"],
    "from": ["F", "R", "AH", "M"],
    "good": ["G", "UH", "D"],
    "hello": ["HH", "AH", "L", "OW"],
    "hi": ["HH", "AY"],
    "how": ["HH", "AW"],
    "is": ["IH", "Z"],
    "language": ["L", "AE", "NG", "G", "W", "IH", "JH"],
    "model": ["M", "AA", "D", "AH", "L"],
    "morning": ["M", "AO", "R", "N", "IH", "NG"],
    "my": ["M", "AY"],
    "name": ["N", "EY", "M"],
    "no": ["N", "OW"],
    "of": ["AH", "V"],
    "one": ["W", "AH", "N"],
    "open": ["OW", "P", "AH", "N"],
    "paper": ["P", "EY", "P", "ER"],
    "please": ["P", "L", "IY", "Z"],
    "prosody": ["P", "R", "AA", "S", "AH", "D", "IY"],
    "research": ["R", "IY", "S", "ER", "CH"],
    "sound": ["S", "AW", "N", "D"],
    "source": ["S", "AO", "R", "S"],
    "speaker": ["S", "P", "IY", "K", "ER"],
    "speech": ["S", "P", "IY", "CH"],
    "synthesis": ["S", "IH", "N", "TH", "AH", "S", "IH", "S"],
    "test": ["T", "EH", "S", "T"],
    "thank": ["TH", "AE", "NG", "K"],
    "the": ["DH", "AH"],
    "this": ["DH", "IH", "S"],
    "three": ["TH", "R", "IY"],
    "timbre": ["T", "AE", "M", "B", "ER"],
    "to": ["T", "UW"],
    "today": ["T", "AH", "D", "EY"],
    "two": ["T", "UW"],
    "voice": ["V", "OY", "S"],
    "we": ["W", "IY"],
    "welcome": ["W", "EH", "L", "K", "AH", "M"],
    "world": ["W", "ER", "L", "D"],
    "yes": ["Y", "EH", "S"],
    "you": ["Y", "UW"],
}
