from timbrel.cli import main


def test_phonemize_command(capsys):
    rc = main(["phonemize", "你好", "-l", "zh"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "n" in out.split()


def test_info_command(capsys):
    rc = main(["info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "timbrel" in out
    assert "vocab_size=" in out


def test_symbols_command(capsys):
    rc = main(["symbols"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "zh" in out.split()


def test_no_command_prints_help_and_returns_nonzero(capsys):
    rc = main([])
    assert rc == 1
