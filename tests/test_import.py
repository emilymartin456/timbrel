import timbrel


def test_package_exposes_version():
    assert isinstance(timbrel.__version__, str)
    assert timbrel.__version__.count(".") >= 2
