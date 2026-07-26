"""Models used by the evaluation scripts.

Deliberately free of batch normalisation: running statistics are buffers, and
buffer updates do not propagate back out of ``functional_call`` under ``vmap``,
so a batch-normalised member would silently train on stale statistics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from optuna_saturate.vectorized.dropout import member_dropout

IMAGE_SIZE = 28
CHANNELS = 1
CLASSES = 10


class SmallCNN(nn.Module):
    """Two convolutional blocks and a classifier head.

    ``width`` scales the channel count and therefore every tensor shape, so it is
    a shape-changing hyperparameter and cannot be vectorised. Dropout arrives per
    member through ``hp`` and is vectorisable.

    Args:
        width: Channels in the first block; the second block doubles it.
        dropout_default: Probability used when ``hp`` carries none.
    """

    def __init__(self, width: int = 16, dropout_default: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        side = IMAGE_SIZE // 4
        self.fc = nn.Linear(width * 2 * side * side, CLASSES)
        self.dropout_default = dropout_default

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        h = self.pool(torch.relu(self.conv1(x)))
        h = self.pool(torch.relu(self.conv2(h)))
        h = h.flatten(1)
        if hp is not None and "dropout" in hp:
            h = member_dropout(h, p=hp["dropout"], training=self.training)
        elif self.dropout_default > 0.0:
            h = member_dropout(
                h,
                p=torch.tensor(self.dropout_default, device=h.device),
                training=self.training,
            )
        return self.fc(h)


def build_cnn_members(k: int, width: int, seed: int = 0) -> list[SmallCNN]:
    """Build ``k`` independently initialised networks with reproducible weights."""
    members = []
    for i in range(k):
        torch.manual_seed(seed + i)
        members.append(SmallCNN(width=width))
    return members


def parameter_count(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())
