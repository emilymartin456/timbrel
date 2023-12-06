"""Phoneme inventory for the bilingual (Mandarin + English) front-end.

The inventory keeps the two languages disjoint by convention:

* Mandarin symbols are lower-case. Initials are bare consonants (``zh``, ``b``);
  finals carry a trailing tone digit 1-5 (``ong1``, ``a4``, ``e5``).
* English symbols are upper-case ARPAbet phones without stress (``AA``, ``ZH``).

Because tone digits and letter-case never overlap, a single flat vocabulary can
hold both languages without collisions.
"""

from __future__ import annotations

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SIL = "sil"  # long/utterance silence
SP = "sp"  # short pause (comma-level)

SPECIALS = [PAD, UNK, BOS, EOS, SIL, SP]

# 23 Mandarin initials (声母). ``y``/``w`` are included as glide onsets.
MANDARIN_INITIALS = [
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s",
    "y", "w",
]

# Base finals (韵母). Written the way pypinyin renders them (``iu`` not ``iou``).
MANDARIN_FINAL_BASES = [
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong", "er",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng",
    "ve", "van", "vn", "io",
]

TONES = ["1", "2", "3", "4", "5"]  # 5 == neutral tone (轻声)

MANDARIN_FINALS = [base + tone for base in MANDARIN_FINAL_BASES for tone in TONES]

# ARPAbet phones (stress markers dropped).
ENGLISH_PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]


def build_symbols() -> list[str]:
    """Return the full ordered symbol list (specials first)."""
    return SPECIALS + MANDARIN_INITIALS + MANDARIN_FINALS + ENGLISH_PHONEMES


class PhonemeVocab:
    """Bidirectional mapping between phoneme symbols and integer ids."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = symbols if symbols is not None else build_symbols()
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("phoneme inventory contains duplicate symbols")
        self._sym2id = {s: i for i, s in enumerate(self.symbols)}

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._sym2id

    @property
    def pad_id(self) -> int:
        return self._sym2id[PAD]

    @property
    def unk_id(self) -> int:
        return self._sym2id[UNK]

    @property
    def bos_id(self) -> int:
        return self._sym2id[BOS]

    @property
    def eos_id(self) -> int:
        return self._sym2id[EOS]

    def encode(self, phonemes: list[str], add_bos_eos: bool = False) -> list[int]:
        ids = [self._sym2id.get(p, self.unk_id) for p in phonemes]
        if add_bos_eos:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> list[str]:
        return [self.symbols[i] for i in ids]
