"""Stack K identically shaped models into one set of batched parameters.

``torch.func.stack_module_state`` turns K module instances into a dictionary of
tensors carrying a leading member dimension. ``functional_call`` under ``vmap``
then evaluates all K members in a single pass, which is what keeps a small model
from leaving the device idle.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call, stack_module_state, vmap


class StackedEnsemble:
    """K models of identical architecture evaluated as one batched model.

    Every member model must accept the forward signature ``forward(x, hp=None)``,
    where ``hp`` maps a hyperparameter name to a zero-dimensional tensor holding
    that member's value. Models that need no per-member hyperparameters simply
    ignore the argument.

    Attributes:
        k: Number of stacked members.
        params: Stacked parameters, each of shape ``[k, *original_shape]``.
        buffers: Stacked buffers, same convention.
    """

    def __init__(self, models: Sequence[nn.Module]) -> None:
        if len(models) == 0:
            raise ValueError("StackedEnsemble needs at least one model")

        self.k = len(models)

        stacked_params, stacked_buffers = stack_module_state(list(models))
        # stack_module_state returns detached tensors; make them differentiable
        # leaves so an ordinary autograd backward pass populates .grad.
        self.params: dict[str, Tensor] = {
            name: tensor.detach().clone().requires_grad_(True)
            for name, tensor in stacked_params.items()
        }
        self.buffers: dict[str, Tensor] = {
            name: tensor.detach().clone() for name, tensor in stacked_buffers.items()
        }

        # A meta-device copy supplies the module structure without allocating
        # storage — every value comes from params/buffers at call time.
        self._base = copy.deepcopy(models[0]).to("meta")

    def forward(self, x: Tensor, member_hp: dict[str, Tensor] | None = None) -> Tensor:
        """Evaluate every member on the shared batch ``x``.

        Args:
            x: Input batch shared by all members, shape ``[batch, ...]``.
            member_hp: Optional per-member hyperparameters. Every value must be a
                tensor of shape ``[k]``; member ``i`` receives element ``i`` as a
                zero-dimensional tensor.

        Returns:
            Outputs of shape ``[k, batch, ...]``.
        """
        for name, tensor in (member_hp or {}).items():
            if tensor.shape != (self.k,):
                raise ValueError(
                    f"member hyperparameter {name!r} must have shape ({self.k},), "
                    f"got {tuple(tensor.shape)}"
                )

        # randomness="different" gives each member its own random draws, which is
        # required for per-member dropout masks.
        if member_hp:

            def call(
                params: dict[str, Tensor],
                buffers: dict[str, Tensor],
                inputs: Tensor,
                member: dict[str, Tensor],
            ) -> Tensor:
                out: Tensor = functional_call(self._base, (params, buffers), (inputs, member))
                return out

            batched = vmap(call, in_dims=(0, 0, None, 0), randomness="different")
            result: Tensor = batched(self.params, self.buffers, x, member_hp)
            return result

        # An empty hp dict cannot be vmapped over — it carries no batched tensor
        # to take the member dimension from — so it is dropped from the signature.
        def call_without_hp(
            params: dict[str, Tensor],
            buffers: dict[str, Tensor],
            inputs: Tensor,
        ) -> Tensor:
            out: Tensor = functional_call(self._base, (params, buffers), (inputs, None))
            return out

        plain = vmap(call_without_hp, in_dims=(0, 0, None), randomness="different")
        plain_result: Tensor = plain(self.params, self.buffers, x)
        return plain_result

    def zero_grad(self) -> None:
        """Drop gradients accumulated on the stacked parameters."""
        for tensor in self.params.values():
            tensor.grad = None

    def member_state_dict(self, index: int) -> dict[str, Tensor]:
        """Extract one member's parameters as an ordinary state dict."""
        if not 0 <= index < self.k:
            raise IndexError(f"member index {index} out of range for k={self.k}")
        with torch.no_grad():
            return {name: tensor[index].detach().clone() for name, tensor in self.params.items()}
