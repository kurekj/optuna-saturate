"""The handle an objective uses while its group trains as one batched model.

Everything here forwards to the vectorisation layer. The context owns the
device, so the user never writes ``.to(device)`` for anything the library handed
them, and checks that the members share an architecture before stacking them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn

from optuna_saturate.exceptions import ShapeChangingHyperparameterError
from optuna_saturate.vectorized.ensemble import StackedEnsemble
from optuna_saturate.vectorized.loop import train_step
from optuna_saturate.vectorized.optim import (
    VectorizedAdam,
    VectorizedAdamW,
    VectorizedOptimizer,
    VectorizedSGD,
)


def _shape_disagreements(models: Sequence[nn.Module]) -> dict[str, list[tuple[int, ...]]]:
    """Parameter names whose shape is not the same across every member."""
    reference = dict(models[0].named_parameters())
    disagreements: dict[str, list[tuple[int, ...]]] = {}
    for name, tensor in reference.items():
        shapes = [tuple(tensor.shape)]
        for other in models[1:]:
            found = dict(other.named_parameters()).get(name)
            if found is None or tuple(found.shape) != shapes[0]:
                shapes.append(() if found is None else tuple(found.shape))
        if len(set(shapes)) > 1:
            disagreements[name] = shapes
    return disagreements


class VectorizedContext:
    """Primitives for training K configurations as one model.

    Args:
        k: Group size. Every helper assumes exactly this many members.
        declared: Hyperparameter names the user declared vectorisable. Used only
            to write a useful message when the declaration turns out to be wrong.
        device: Where models and tensors live. ``None`` means the default device.
        autocast_dtype: Precision for the forward pass and the loss in
            :meth:`step`, for example ``torch.bfloat16``. ``None`` keeps float32.
    """

    def __init__(
        self,
        k: int,
        declared: tuple[str, ...],
        device: str | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        self.k = k
        self._declared = declared
        self._device = device
        self.autocast_dtype = autocast_dtype

    def stack(self, models: Sequence[nn.Module]) -> StackedEnsemble:
        """Flatten K members into one batched model.

        Raises:
            ValueError: ``models`` does not hold exactly ``k`` members.
            ShapeChangingHyperparameterError: The members disagree on a parameter
                shape, which means something in ``declared`` changes the
                architecture and cannot be vectorised.
        """
        if len(models) != self.k:
            raise ValueError(f"expected {self.k} members to stack, got {len(models)}")

        disagreements = _shape_disagreements(models)
        if disagreements:
            detail = "; ".join(
                f"{name}: {' vs '.join(str(shape) for shape in shapes)}"
                for name, shapes in sorted(disagreements.items())
            )
            raise ShapeChangingHyperparameterError(
                "members of a vectorised group must share every parameter shape, but "
                f"these differ -- {detail}. One of the hyperparameters declared "
                f"vectorisable ({', '.join(self._declared) or 'none'}) changes the "
                "architecture. Remove it from `over` so it runs as its own trial."
            )

        placed = [model.to(self._device) if self._device else model for model in models]
        return StackedEnsemble(placed)

    def sgd(
        self,
        ensemble: StackedEnsemble,
        lr: float | Tensor,
        momentum: float | Tensor = 0.0,
        weight_decay: float | Tensor = 0.0,
        nesterov: bool = False,
    ) -> VectorizedSGD:
        """Build an SGD optimiser applying one value of each hyperparameter per member.

        Every hyperparameter accepts either a float shared by the group or the
        ``[k]`` tensor the trial proxy returns for a vectorised name.
        """
        return VectorizedSGD(
            ensemble.params,
            lr=self._place_hp(lr),
            momentum=self._place_hp(momentum),
            weight_decay=self._place_hp(weight_decay),
            nesterov=nesterov,
        )

    def adam(
        self,
        ensemble: StackedEnsemble,
        lr: float | Tensor,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float | Tensor = 0.0,
    ) -> VectorizedAdam:
        """Build an Adam optimiser applying one value of each hyperparameter per member."""
        return VectorizedAdam(
            ensemble.params,
            lr=self._place_hp(lr),
            betas=(self._place_hp(betas[0]), self._place_hp(betas[1])),
            eps=eps,
            weight_decay=self._place_hp(weight_decay),
        )

    def adamw(
        self,
        ensemble: StackedEnsemble,
        lr: float | Tensor,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float | Tensor | None = None,
    ) -> VectorizedAdamW:
        """Build an AdamW optimiser applying one value of each hyperparameter per member.

        ``weight_decay`` defaults to ``0.01``, as it does in ``torch.optim.AdamW``.
        """
        return VectorizedAdamW(
            ensemble.params,
            lr=self._place_hp(lr),
            betas=(self._place_hp(betas[0]), self._place_hp(betas[1])),
            eps=eps,
            weight_decay=None if weight_decay is None else self._place_hp(weight_decay),
        )

    def step(
        self,
        ensemble: StackedEnsemble,
        optimiser: VectorizedOptimizer,
        x: Tensor,
        y: Tensor,
        member_hp: dict[str, Tensor] | None = None,
        temperature: Tensor | None = None,
    ) -> Tensor:
        """Run one optimisation step for every member at once.

        Returns:
            Losses of shape ``[k]``, detached.
        """
        inputs, targets = self._place(x), self._place(y)
        placed_hp = (
            {name: self._place(value) for name, value in member_hp.items()} if member_hp else None
        )
        placed_temperature = self._place(temperature) if temperature is not None else None

        return train_step(
            ensemble,
            optimiser,
            inputs,
            targets,
            temperature=placed_temperature,
            member_hp=placed_hp,
            autocast_dtype=self.autocast_dtype,
        )

    def accuracy(
        self,
        ensemble: StackedEnsemble,
        batches: Iterable[tuple[Tensor, Tensor]],
    ) -> list[float]:
        """Top-1 accuracy of every member over ``batches``.

        Returns:
            ``k`` plain floats, ready to hand straight back to Optuna.
        """
        correct = torch.zeros(self.k, device=self._device)
        total = 0
        was_training = ensemble.training
        ensemble.eval()  # dropout off, as model.eval() would do for a single model
        try:
            with torch.no_grad():
                for x, y in batches:
                    inputs, targets = self._place(x), self._place(y)
                    predictions = ensemble.forward(inputs).argmax(dim=-1)
                    correct += (predictions == targets.unsqueeze(0)).sum(dim=1)
                    total += int(targets.shape[0])
        finally:
            ensemble.train(was_training)
        if total == 0:
            raise ValueError("cannot compute accuracy over an empty set of batches")
        scores: list[float] = (correct / total).tolist()
        return scores

    def _place(self, tensor: Tensor) -> Tensor:
        return tensor.to(self._device) if self._device else tensor

    def _place_hp(self, value: float | Tensor) -> float | Tensor:
        return self._place(value) if isinstance(value, Tensor) else value
