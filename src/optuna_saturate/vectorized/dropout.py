"""Dropout with a per-member probability.

``torch.nn.functional.dropout`` takes ``p`` as a Python float, which fixes one
rate for the whole call. Inside ``vmap`` each member carries its own probability
as a zero-dimensional tensor, so the Bernoulli mask has to be built by hand.
"""

from __future__ import annotations

import torch
from torch import Tensor


def validate_drop_probability(p: Tensor, name: str = "dropout") -> None:
    """Check that every member's drop probability lies in ``[0, 1)``.

    Call this on the full ``[k]`` vector **before** entering ``vmap``. Inside
    ``vmap`` the tensor is batched and cannot be converted to a Python bool, so
    the check is impossible there.

    Args:
        p: Drop probabilities, one per member.
        name: Hyperparameter name, used in the error message.

    Raises:
        ValueError: Some probability is negative or greater than or equal to 1.
            A probability of exactly 1 would zero every activation and make the
            inverted-dropout rescaling divide by zero.
    """
    if bool(torch.any(p < 0.0)) or bool(torch.any(p >= 1.0)):
        raise ValueError(f"{name} probability must lie in [0, 1), got {p.tolist()}")


def member_dropout(x: Tensor, p: Tensor, training: bool) -> Tensor:
    """Apply inverted dropout using a probability held in a tensor.

    This is pure arithmetic with no validation: it runs inside ``vmap``, where
    ``p`` is a batched tensor and any branch on its value would raise. Validate
    with :func:`validate_drop_probability` before the vectorised call.

    Args:
        x: Activations to mask.
        p: Drop probability. Inside ``vmap`` this is the current member's own
            zero-dimensional slice of a ``[k]`` tensor. Must lie in ``[0, 1)``.
        training: When ``False`` the input passes through untouched, matching the
            behaviour of ``nn.Dropout`` in evaluation mode.

    Returns:
        Masked activations, rescaled by ``1 / (1 - p)`` so the expected value is
        preserved and no rescaling is needed at evaluation time.
    """
    if not training:
        return x

    keep = 1.0 - p
    mask = (torch.rand_like(x) < keep).to(x.dtype)
    return x * mask / keep
