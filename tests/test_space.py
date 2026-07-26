import pytest

from optuna_saturate.core.space import SpaceClassification, classify
from optuna_saturate.exceptions import UnknownHyperparameterError


def test_classify_splits_declared_names_from_the_rest() -> None:
    result = classify(
        param_names=["learning_rate", "dropout", "hidden_dim", "n_layers"],
        declared_preserving=["learning_rate", "dropout"],
    )
    assert result == SpaceClassification(
        preserving=("dropout", "learning_rate"),
        changing=("hidden_dim", "n_layers"),
    )


def test_classify_returns_sorted_names_for_determinism() -> None:
    result = classify(
        param_names=["b", "a", "d", "c"],
        declared_preserving=["d", "a"],
    )
    assert result.preserving == ("a", "d")
    assert result.changing == ("b", "c")


def test_classify_with_nothing_declared_makes_everything_shape_changing() -> None:
    result = classify(param_names=["lr", "width"], declared_preserving=[])
    assert result.preserving == ()
    assert result.changing == ("lr", "width")


def test_classify_rejects_a_declared_name_that_is_not_in_the_space() -> None:
    with pytest.raises(UnknownHyperparameterError) as excinfo:
        classify(param_names=["learning_rate"], declared_preserving=["lerning_rate"])
    assert "lerning_rate" in str(excinfo.value)
