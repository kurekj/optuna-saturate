"""Loss functions that reduce per stacked member instead of over the whole batch.

The ensemble emits logits of shape ``[k, batch, classes]``. Reducing the whole
tensor at once would average across members and destroy the per-member signal, so
these functions reduce over the batch only and return one value per member.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def per_member_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    temperature: Tensor | None = None,
) -> Tensor:
    """Cross-entropy reduced over the batch, kept separate per member.

    Args:
        logits: Ensemble outputs of shape ``[k, batch, classes]``.
        targets: Class indices, either ``[batch]`` when every member sees the same
            targets, or ``[k, batch]`` for per-member targets.
        temperature: Optional softmax temperatures of shape ``[k]``. Logits of
            member ``i`` are divided by ``temperature[i]`` before the softmax.

    Returns:
        Losses of shape ``[k]``. Call ``.sum().backward()`` on the result: members
        share no parameters, so the sum's gradient with respect to member ``i``
        equals the gradient of member ``i``'s own loss.

    Raises:
        ValueError: ``temperature`` has a length other than the member count.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must have shape [k, batch, classes], got {tuple(logits.shape)}")

    k = logits.shape[0]

    if temperature is not None:
        if temperature.shape != (k,):
            raise ValueError(f"temperature must have shape ({k},), got {tuple(temperature.shape)}")
        logits = logits / temperature.view(k, 1, 1)

    if targets.dim() == 1:
        targets = targets.unsqueeze(0).expand(k, -1)
    elif targets.shape[0] != k:
        raise ValueError(
            f"per-member targets must have shape ({k}, batch), got {tuple(targets.shape)}"
        )

    losses = [F.cross_entropy(logits[i], targets[i]) for i in range(k)]
    return torch.stack(losses)
