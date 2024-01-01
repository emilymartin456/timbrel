from timbrel.frontend.chinese import ChineseG2P


def test_single_syllable_initial_final():
    g = ChineseG2P()
    assert g.to_phonemes("中") == ["zh", "ong1"]


def test_labial_glide_initial():
    g = ChineseG2P()
    assert g.to_phonemes("文") == ["w", "en2"]


def test_zero_initial_syllable():
    g = ChineseG2P()
    assert g.to_phonemes("爱") == ["ai4"]


def test_third_tone_sandhi():
    g = ChineseG2P()
    # 你好: ni3 hao3 -> the first third tone becomes a rising (2) tone
    phones = g.to_phonemes("你好")
    assert phones == ["n", "i2", "h", "ao3"]


def test_punctuation_becomes_pause():
    g = ChineseG2P()
    phones = g.to_phonemes("好。")
    assert phones[-1] == "sil"
