import pytest
import torch

from optuna_saturate.vectorized.dropout import member_dropout, validate_drop_probability
from optuna_saturate.vectorized.ensemble import StackedEnsemble
from tests.models import build_dropout_members


def test_evaluation_mode_returns_the_input_unchanged() -> None:
    x = torch.randn(100)
    out = member_dropout(x, p=torch.tensor(0.5), training=False)
    torch.testing.assert_close(out, x)


def test_probability_zero_keeps_every_element() -> None:
    x = torch.randn(1000)
    out = member_dropout(x, p=torch.tensor(0.0), training=True)
    torch.testing.assert_close(out, x)


def test_the_fraction_of_dropped_elements_approximates_the_probability() -> None:
    torch.manual_seed(0)
    x = torch.ones(200_000)

    out = member_dropout(x, p=torch.tensor(0.3), training=True)

    dropped = (out == 0).to(torch.float32).mean().item()
    assert 0.29 < dropped < 0.31


def test_inverted_scaling_preserves_the_expected_value() -> None:
    torch.manual_seed(0)
    x = torch.ones(200_000)

    out = member_dropout(x, p=torch.tensor(0.4), training=True)

    assert 0.98 < out.mean().item() < 1.02


def test_validator_rejects_a_probability_of_one() -> None:
    with pytest.raises(ValueError, match="probability"):
        validate_drop_probability(torch.tensor([0.1, 1.0]))


def test_validator_rejects_a_negative_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        validate_drop_probability(torch.tensor([-0.2, 0.3]))


def test_validator_accepts_the_whole_half_open_unit_interval() -> None:
    validate_drop_probability(torch.tensor([0.0, 0.5, 0.999]))  # must not raise


def test_validator_names_the_offending_hyperparameter() -> None:
    with pytest.raises(ValueError, match="feature_dropout"):
        validate_drop_probability(torch.tensor([1.5]), name="feature_dropout")


def test_each_member_drops_at_its_own_rate_inside_the_ensemble() -> None:
    torch.manual_seed(0)
    members = build_dropout_members(k=3, in_features=8, hidden=512, out_features=4)
    ensemble = StackedEnsemble(members)
    x = torch.randn(64, 8)

    out = ensemble.forward(x, member_hp={"dropout": torch.tensor([0.0, 0.5, 0.9])})
    # Omitting member_hp disables dropout, giving the deterministic reference.
    reference = ensemble.forward(x)

    # The dropped units sit in the hidden layer, and the final linear layer mixes
    # them back into dense values that are never exactly zero. So the rate shows
    # up as how far a member strays from its own dropout-free output, not as a
    # count of zeros. A member at p=0 must not stray at all.
    deviation = [(out[i] - reference[i]).abs().mean().item() for i in range(3)]
    assert deviation[0] == 0.0
    assert deviation[0] < deviation[1] < deviation[2]


def test_members_receive_independent_random_masks() -> None:
    """With randomness='different' two members at the same rate must not coincide."""
    torch.manual_seed(0)
    members = build_dropout_members(k=2, in_features=8, hidden=512, out_features=4)
    # Identical weights isolate the mask as the only source of difference.
    members[1].load_state_dict(members[0].state_dict())
    ensemble = StackedEnsemble(members)
    x = torch.randn(64, 8)

    out = ensemble.forward(x, member_hp={"dropout": torch.tensor([0.5, 0.5])})

    assert not torch.allclose(out[0], out[1])
