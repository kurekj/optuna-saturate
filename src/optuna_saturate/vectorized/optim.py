"""Optimisers that apply a different learning rate to each stacked member.

Stacked parameters carry a leading member dimension, so a single tensor holds K
members at once. ``torch.optim`` applies one scalar learning rate per parameter
group and cannot address members separately. These optimisers reshape a learning
rate vector of shape ``[k]`` to ``[k, 1, 1, ...]`` and broadcast it over the
gradient instead.
"""

from __future__ import annotations

import torch
from torch import Tensor


class VectorizedSGD:
    """Plain stochastic gradient descent with one learning rate per member.

    Args:
        params: Stacked parameters, each of shape ``[k, *original_shape]``.
        lr: Learning rates of shape ``[k]``.
    """

    def __init__(self, params: dict[str, Tensor], lr: Tensor) -> None:
        if lr.dim() != 1:
            raise ValueError(f"learning rate must be one-dimensional, got shape {tuple(lr.shape)}")

        member_counts = {tensor.shape[0] for tensor in params.values()}
        if len(member_counts) > 1:
            raise ValueError(
                f"stacked parameters disagree on the member count: {sorted(member_counts)}"
            )
        k = member_counts.pop() if member_counts else 0
        if lr.shape[0] != k:
            raise ValueError(
                f"learning rate vector has length {lr.shape[0]} but there are {k} members"
            )

        self.params = params
        self.lr = lr

    def step(self) -> None:
        """Apply one descent step to every member that has a gradient."""
        with torch.no_grad():
            for tensor in self.params.values():
                grad = tensor.grad
                if grad is None:
                    continue
                # Reshape [k] to [k, 1, 1, ...] so it broadcasts over the member's
                # own dimensions without touching the member axis itself.
                lr = self.lr.view(-1, *([1] * (tensor.dim() - 1)))
                tensor.sub_(lr * grad)

    def zero_grad(self) -> None:
        """Drop gradients accumulated on the stacked parameters."""
        for tensor in self.params.values():
            tensor.grad = None
