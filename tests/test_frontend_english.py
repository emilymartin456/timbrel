from timbrel.frontend.english import EnglishG2P


def test_lexicon_lookup():
    g = EnglishG2P()
    assert g.to_phonemes("hello") == ["HH", "AH", "L", "OW"]
    assert g.to_phonemes("world") == ["W", "ER", "L", "D"]


def test_all_symbols_are_arpabet_or_pause():
    g = EnglishG2P()
    phones = g.to_phonemes("hello, world.")
    for p in phones:
        assert p.isupper() or p in ("sp", "sil")


def test_punctuation_maps_to_pause():
    g = EnglishG2P()
    phones = g.to_phonemes("yes, no.")
    assert "sp" in phones
    assert phones[-1] == "sil"


def test_oov_falls_back_to_rules():
    g = EnglishG2P()
    phones = g.to_phonemes("blorp")
    assert phones  # non-empty
    assert all(p.isupper() for p in phones)
