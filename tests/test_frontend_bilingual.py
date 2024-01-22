from timbrel.frontend.bilingual import BilingualFrontend


def test_pure_chinese():
    f = BilingualFrontend()
    assert f.to_phonemes("你好", language="zh") == ["n", "i2", "h", "ao3"]


def test_pure_english():
    f = BilingualFrontend()
    assert f.to_phonemes("hello world", language="en") == [
        "HH", "AH", "L", "OW", "W", "ER", "L", "D",
    ]


def test_code_switch_auto():
    f = BilingualFrontend()
    phones = f.to_phonemes("你好 world")
    assert "n" in phones  # from 你
    assert "W" in phones  # from world


def test_encode_matches_phoneme_count():
    f = BilingualFrontend()
    text = "你好 world"
    assert len(f.encode(text)) == len(f.to_phonemes(text))


def test_empty_and_whitespace_input():
    f = BilingualFrontend()
    assert f.to_phonemes("") == []
    assert f.to_phonemes("   ") == []


def test_no_dangling_edge_pause():
    f = BilingualFrontend()
    phones = f.to_phonemes("。你好。")
    assert phones[0] not in ("sp", "sil")
    assert phones[-1] not in ("sp", "sil")
