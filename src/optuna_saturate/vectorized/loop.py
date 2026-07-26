"""The vectorised training loop: one pass trains every member of a group."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.losses import per_member_cross_entropy
from optuna_saturate.vectorized.optim import VectorizedSGD


def train_step(
    ensemble: StackedEnsemble,
    optimiser: VectorizedSGD,
    inputs: Tensor,
    targets: Tensor,
    temperature: Tensor | None = None,
    member_hp: dict[str, Tensor] | None = None,
) -> Tensor:
    """Run one optimisation step, training every member simultaneously.

    Args:
        ensemble: The stacked models to train.
        optimiser: Optimiser holding the same parameter dictionary as ``ensemble``.
        inputs: One batch of inputs, shared by every member.
        targets: The matching targets.
        temperature: Optional per-member softmax temperatures of shape ``[k]``.
        member_hp: Optional per-member hyperparameters forwarded to the models.

    Returns:
        Losses of shape ``[k]``, detached from the graph.
    """
    optimiser.zero_grad()
    logits = ensemble.forward(inputs, member_hp=member_hp)
    losses = per_member_cross_entropy(logits, targets, temperature=temperature)
    # Summing before backward is what makes one pass train every member; see
    # per_member_cross_entropy for why the gradients stay separate.
    # PyTorch ships Tensor.backward without annotations, hence the ignore.
    losses.sum().backward()  # type: ignore[no-untyped-call]
    optimiser.step()
    return losses.detach()


def train_steps(
    ensemble: StackedEnsemble,
    optimiser: VectorizedSGD,
    batches: Sequence[tuple[Tensor, Tensor]],
    temperature: Tensor | None = None,
    member_hp: dict[str, Tensor] | None = None,
) -> list[Tensor]:
    """Run :func:`train_step` once per batch and keep every loss.

    Returns:
        One loss tensor of shape ``[k]`` per batch. Each is cloned, so the
        history does not alias a buffer a later step may overwrite.
    """
    return [
        train_step(
            ensemble, optimiser, inputs, targets,
            temperature=temperature, member_hp=member_hp,
        ).clone()
        for inputs, targets in batches
    ]
