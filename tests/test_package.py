import optuna_saturate


def test_package_exposes_a_version() -> None:
    assert isinstance(optuna_saturate.__version__, str)
    assert optuna_saturate.__version__ != ""
