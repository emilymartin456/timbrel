from timbrel.frontend import PhonemeVocab, build_symbols


def test_specials_come_first():
    v = PhonemeVocab()
    assert v.pad_id == 0
    assert v.unk_id == 1
    assert len(v) > 200


def test_no_duplicate_symbols():
    syms = build_symbols()
    assert len(syms) == len(set(syms))


def test_encode_decode_roundtrip_bilingual():
    v = PhonemeVocab()
    syms = ["zh", "ong1", "W", "ER"]  # mandarin initial+final then english phones
    ids = v.encode(syms)
    assert all(isinstance(i, int) for i in ids)
    assert v.decode(ids) == syms


def test_unknown_symbol_maps_to_unk():
    v = PhonemeVocab()
    assert v.encode(["definitely-not-a-phone"]) == [v.unk_id]


def test_add_bos_eos():
    v = PhonemeVocab()
    ids = v.encode(["a1"], add_bos_eos=True)
    assert ids[0] == v.bos_id
    assert ids[-1] == v.eos_id
