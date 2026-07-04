from timbrel.config import Config, ModelConfig, load_config, save_config


def test_defaults_are_consistent():
    c = Config()
    assert c.model.n_mels == c.audio.n_mels == 80
    assert c.train.betas == [0.9, 0.98]


def test_yaml_roundtrip(tmp_path):
    c = Config()
    c.model.hidden = 192
    c.train.lr = 3e-4
    path = tmp_path / "cfg.yaml"
    save_config(c, path)
    loaded = load_config(path)
    assert loaded.model.hidden == 192
    assert loaded.train.lr == 3e-4
    assert loaded.to_dict() == c.to_dict()


def test_from_dict_partial_uses_defaults():
    c = Config.from_dict({"model": {"hidden": 64}})
    assert c.model.hidden == 64
    assert c.audio.sample_rate == 22050


def test_model_config_standalone():
    m = ModelConfig(hidden=128, n_heads=4)
    assert m.hidden == 128
    assert m.n_heads == 4
