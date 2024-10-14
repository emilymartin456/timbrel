"""Bilingual (Mandarin + English) front-end.

Routes each run of a code-switched string to the right g2p and returns either a
flat list of phoneme symbols or their integer ids under a :class:`PhonemeVocab`.
"""

from __future__ import annotations

from timbrel.config import FrontendConfig
from timbrel.frontend.chinese import ChineseG2P
from timbrel.frontend.english import EnglishG2P
from timbrel.frontend.normalize import normalize
from timbrel.frontend.phonemes import SIL, SP, PhonemeVocab
from timbrel.frontend.segment import segment

_STOP_PUNCT = set("。！？!?.…")
_PAUSE_PUNCT = set("，、,;:；：")


def _collapse_pauses(phones: list[str]) -> list[str]:
    """Merge runs of adjacent pauses, keeping the stronger (``sil``) one."""
    out: list[str] = []
    for p in phones:
        if p in (SP, SIL) and out and out[-1] in (SP, SIL):
            if p == SIL:
                out[-1] = SIL
            continue
        out.append(p)
    return out


def _strip_edge_pauses(phones: list[str]) -> list[str]:
    """Drop leading/trailing pause tokens so an utterance never starts on one."""
    start, end = 0, len(phones)
    while start < end and phones[start] in (SP, SIL):
        start += 1
    while end > start and phones[end - 1] in (SP, SIL):
        end -= 1
    return phones[start:end]


class BilingualFrontend:
    """Turn arbitrary zh/en text into phonemes (symbols or ids)."""

    def __init__(
        self,
        vocab: PhonemeVocab | None = None,
        config: FrontendConfig | None = None,
    ) -> None:
        self.config = config or FrontendConfig()
        self.vocab = vocab or PhonemeVocab()
        pause = self.config.insert_pause_on_punct
        self.zh = ChineseG2P(insert_pause=pause)
        self.en = EnglishG2P(insert_pause=pause)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _punct_to_pause(self, chunk: str) -> list[str]:
        out: list[str] = []
        for ch in chunk:
            if ch in _STOP_PUNCT:
                out.append(SIL)
            elif ch in _PAUSE_PUNCT:
                out.append(SP)
        return out

    def to_phonemes(self, text: str, language: str | None = None) -> list[str]:
        if not text or not text.strip():
            return []
        language = language or self.config.language
        if language in ("zh", "en"):
            text = normalize(text, language)
            g2p = self.zh if language == "zh" else self.en
            phones = g2p.to_phonemes(text)
        else:
            text = normalize(text, "auto")
            phones = []
            for lang, chunk in segment(text):
                if lang == "zh":
                    phones.extend(self.zh.to_phonemes(chunk))
                elif lang == "en":
                    phones.extend(self.en.to_phonemes(chunk))
                else:
                    phones.extend(self._punct_to_pause(chunk))
        return _strip_edge_pauses(_collapse_pauses(phones))

    def encode(
        self,
        text: str,
        language: str | None = None,
        add_bos_eos: bool | None = None,
    ) -> list[int]:
        if add_bos_eos is None:
            add_bos_eos = self.config.add_bos_eos
        phones = self.to_phonemes(text, language)
        return self.vocab.encode(phones, add_bos_eos=add_bos_eos)

    def __call__(self, text: str, language: str | None = None) -> list[int]:
        return self.encode(text, language)
