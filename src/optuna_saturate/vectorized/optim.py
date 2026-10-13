"""Optimisers that apply a different hyperparameter value to each stacked member.

Stacked parameters carry a leading member dimension, so a single tensor holds K
members at once. ``torch.optim`` applies one scalar learning rate per parameter
group and cannot address members separately. The optimisers here accept every
scalar hyperparameter of their ``torch.optim`` counterpart either as a Python
float shared by the group or as a tensor of shape ``[k]`` holding one value per
member, reshape it to ``[k, 1, 1, ...]`` and broadcast it over the gradient.

The update rules follow ``torch.optim.SGD``, ``torch.optim.Adam`` and
``torch.optim.AdamW`` operation for operation, so that a member trained here
reproduces the same member trained alone with the reference optimiser. The parity
tests hold the two paths to ``atol=1e-5`` in float32.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _member_count(params: dict[str, Tensor]) -> int:
    member_counts = {tensor.shape[0] for tensor in params.values()}
    if len(member_counts) > 1:
        raise ValueError(
            f"stacked parameters disagree on the member count: {sorted(member_counts)}"
        )
    return member_counts.pop() if member_counts else 0


class _MemberVector:
    """One hyperparameter, held per member in two precisions.

    ``exact`` keeps the values in float64, which is what ``torch.optim`` works
    with when it raises a beta to the power of the step count; ``cast`` holds
    them in the parameters' own dtype and device for the broadcast update. A
    float given by the caller is shared by every member.
    """

    def __init__(
        self,
        value: float | Tensor,
        k: int,
        name: str,
        like: Tensor | None,
        low: float = 0.0,
        high: float | None = None,
        high_inclusive: bool = True,
    ) -> None:
        if isinstance(value, Tensor):
            if value.dim() != 1:
                raise ValueError(f"{name} must be one-dimensional, got shape {tuple(value.shape)}")
            if value.shape[0] != k:
                raise ValueError(
                    f"{name} vector has length {value.shape[0]} but there are {k} members"
                )
            exact = value.detach().to("cpu", torch.float64)
        else:
            exact = torch.full((k,), float(value), dtype=torch.float64)

        if bool(torch.any(exact < low)):
            raise ValueError(f"{name} must be at least {low}, got {exact.tolist()}")
        if high is not None:
            above = exact > high if high_inclusive else exact >= high
            if bool(torch.any(above)):
                bound = f"at most {high}" if high_inclusive else f"below {high}"
                raise ValueError(f"{name} must be {bound}, got {exact.tolist()}")

        self.name = name
        self.exact = exact
        self.cast = exact.to(like.device, like.dtype) if like is not None else exact.float()

    def broadcast(self, tensor: Tensor) -> Tensor:
        """Reshape ``[k]`` to ``[k, 1, 1, ...]`` so it spans the member axis only."""
        return self.cast.view(-1, *([1] * (tensor.dim() - 1)))

    def any_nonzero(self) -> bool:
        return bool(torch.any(self.exact != 0.0))


class _MemberwiseOptimizer:
    """Shared plumbing: member-count validation, gradient clearing, step counting."""

    def __init__(self, params: dict[str, Tensor]) -> None:
        self.params = params
        self.k = _member_count(params)
        # One count per parameter, advanced only when that parameter receives a
        # gradient, as torch.optim keeps its per-parameter ``step`` state.
        self.steps: dict[str, int] = {}

    def _reference(self) -> Tensor | None:
        return next(iter(self.params.values()), None)

    def zero_grad(self) -> None:
        """Drop gradients accumulated on the stacked parameters."""
        for tensor in self.params.values():
            tensor.grad = None


class VectorizedSGD(_MemberwiseOptimizer):
    """Stochastic gradient descent with one hyperparameter value per member.

    Matches ``torch.optim.SGD`` with ``dampening=0``: with momentum the buffer is
    initialised to the first gradient, and with ``nesterov`` the gradient is
    combined with the momentum buffer before the update.

    Args:
        params: Stacked parameters, each of shape ``[k, *original_shape]``.
        lr: Learning rates of shape ``[k]``, or one float for the whole group.
        momentum: Momentum factor, per member or shared. Zero disables it.
        weight_decay: L2 penalty added to the gradient, per member or shared.
        nesterov: Use Nesterov momentum. Every member then needs a momentum
            above zero, as ``torch.optim.SGD`` requires.
    """

    def __init__(
        self,
        params: dict[str, Tensor],
        lr: float | Tensor,
        momentum: float | Tensor = 0.0,
        weight_decay: float | Tensor = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params)
        like = self._reference()
        self.lr = _MemberVector(lr, self.k, "learning rate", like)
        self.momentum = _MemberVector(momentum, self.k, "momentum", like)
        self.weight_decay = _MemberVector(weight_decay, self.k, "weight decay", like)
        self.nesterov = nesterov
        if nesterov and bool(torch.any(self.momentum.exact <= 0.0)):
            raise ValueError(
                "nesterov momentum requires a momentum above zero for every member, "
                f"got {self.momentum.exact.tolist()}"
            )
        self._buffers: dict[str, Tensor] = {}

    def step(self) -> None:
        """Apply one descent step to every member that has a gradient."""
        uses_momentum = self.momentum.any_nonzero()
        uses_weight_decay = self.weight_decay.any_nonzero()
        with torch.no_grad():
            for name, tensor in self.params.items():
                grad = tensor.grad
                if grad is None:
                    continue
                self.steps[name] = self.steps.get(name, 0) + 1
                lr = self.lr.broadcast(tensor)
                if uses_weight_decay:
                    grad = grad + self.weight_decay.broadcast(tensor) * tensor

                if uses_momentum:
                    momentum = self.momentum.broadcast(tensor)
                    buffer = self._buffers.get(name)
                    if buffer is None:
                        buffer = grad.clone()
                        self._buffers[name] = buffer
                    else:
                        buffer.mul_(momentum).add_(grad)
                    grad = grad + momentum * buffer if self.nesterov else buffer

                tensor.sub_(lr * grad)


class VectorizedAdam(_MemberwiseOptimizer):
    """Adam with one hyperparameter value per member.

    Follows ``torch.optim.Adam`` (``amsgrad=False``), including its bias
    correction, which is computed in float64 exactly as ``torch.optim`` computes
    it in Python floats. ``weight_decay`` is the coupled L2 form that Adam adds
    to the gradient; see :class:`VectorizedAdamW` for the decoupled form.

    Args:
        params: Stacked parameters, each of shape ``[k, *original_shape]``.
        lr: Learning rates of shape ``[k]``, or one float for the whole group.
        betas: Exponential decay rates for the two moment estimates, each in
            ``[0, 1)``. Each may be a float or a ``[k]`` tensor.
        eps: Term added to the denominator for numerical stability.
        weight_decay: L2 penalty, per member or shared. ``None`` selects the
            class default, which is zero here and ``0.01`` for AdamW, as in
            ``torch.optim``.
    """

    decoupled_weight_decay = False
    default_weight_decay = 0.0

    def __init__(
        self,
        params: dict[str, Tensor],
        lr: float | Tensor,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float | Tensor | None = None,
    ) -> None:
        super().__init__(params)
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if weight_decay is None:
            weight_decay = self.default_weight_decay
        like = self._reference()
        self.lr = _MemberVector(lr, self.k, "learning rate", like)
        self.beta1 = _MemberVector(betas[0], self.k, "beta1", like, high=1.0, high_inclusive=False)
        self.beta2 = _MemberVector(betas[1], self.k, "beta2", like, high=1.0, high_inclusive=False)
        self.eps = eps
        self.weight_decay = _MemberVector(weight_decay, self.k, "weight decay", like)
        self._exp_avg: dict[str, Tensor] = {}
        self._exp_avg_sq: dict[str, Tensor] = {}

    def step(self) -> None:
        """Apply one Adam step to every member that has a gradient."""
        uses_weight_decay = self.weight_decay.any_nonzero()
        with torch.no_grad():
            for name, tensor in self.params.items():
                grad = tensor.grad
                if grad is None:
                    continue
                step = self.steps.get(name, 0) + 1
                self.steps[name] = step
                lr = self.lr.broadcast(tensor)
                beta2 = self.beta2.broadcast(tensor)

                if uses_weight_decay:
                    weight_decay = self.weight_decay.broadcast(tensor)
                    if self.decoupled_weight_decay:
                        tensor.mul_(1.0 - lr * weight_decay)
                    else:
                        grad = grad + weight_decay * tensor

                exp_avg = self._exp_avg.get(name)
                exp_avg_sq = self._exp_avg_sq.get(name)
                if exp_avg is None or exp_avg_sq is None:
                    exp_avg = torch.zeros_like(tensor)
                    exp_avg_sq = torch.zeros_like(tensor)
                    self._exp_avg[name] = exp_avg
                    self._exp_avg_sq[name] = exp_avg_sq

                # Same operation order as torch.optim.Adam, so both paths round
                # alike. torch.optim passes ``1 - beta`` as a Python float, that
                # is in float64, and the kernel casts it once; computing it from
                # the float32 copy of beta instead would round differently.
                shape = (-1, *([1] * (tensor.dim() - 1)))
                one_minus_beta1 = (1.0 - self.beta1.exact).to(tensor.device, tensor.dtype)
                one_minus_beta2 = (1.0 - self.beta2.exact).to(tensor.device, tensor.dtype)
                exp_avg.lerp_(grad, one_minus_beta1.view(shape))
                exp_avg_sq.mul_(beta2).add_(one_minus_beta2.view(shape) * grad * grad)

                # Bias corrections from the exact (float64) hyperparameters, as
                # torch.optim computes them in Python floats, then cast once.
                bias_correction1 = 1.0 - self.beta1.exact**step
                bias_correction2 = 1.0 - self.beta2.exact**step
                step_size = (self.lr.exact / bias_correction1).to(tensor.device, tensor.dtype)
                bias_correction2_sqrt = bias_correction2.sqrt().to(tensor.device, tensor.dtype)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt.view(shape)).add_(self.eps)
                # (exp_avg / denom) * step_size, in that order, is what addcdiv_
                # computes in torch.optim; keeping the order keeps the rounding.
                tensor.sub_((exp_avg / denom) * step_size.view(shape))


class VectorizedAdamW(VectorizedAdam):
    """AdamW with one hyperparameter value per member.

    Follows ``torch.optim.AdamW``: weight decay multiplies the parameter by
    ``1 - lr * weight_decay`` before the Adam update instead of entering the
    gradient, and defaults to ``0.01`` as it does there. Arguments are those of
    :class:`VectorizedAdam`.
    """

    decoupled_weight_decay = True
    default_weight_decay = 0.01


VectorizedOptimizer = VectorizedSGD | VectorizedAdam | VectorizedAdamW
