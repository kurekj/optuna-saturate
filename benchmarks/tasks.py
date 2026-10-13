"""The tasks the evaluation runs: a dataset, a model family and a search space.

Three tasks cover three model families on three datasets:

    fashion_cnn   two-block CNN on Fashion-MNIST, plain SGD
                  vectorised: learning_rate, dropout     shared: width
    kmnist_mlp    three-layer MLP on KMNIST, SGD with momentum
                  vectorised: learning_rate, momentum, dropout, weight_decay
                  shared: hidden
    cifar_vit     small Transformer encoder on CIFAR-10 patches, AdamW
                  vectorised: learning_rate, weight_decay, dropout
                  shared: embed_dim

Every task provides the same two objectives, one for ``study.optimize`` and one
for ``saturate``, built from the same model code so that a comparison between
them measures the execution strategy and nothing else. Both objectives seed the
global generator before building the model and again before training, so that a
configuration starts from the same weights and sees the same batch order on
either path; only the dropout masks differ, as they must. The threaded arms
share that generator between trials running at once, which is a property of
threads, not of the objective.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, KMNIST, FashionMNIST

import optuna_saturate as osat
from benchmarks.models import SmallCNN
from optuna_saturate.vectorized.dropout import member_dropout

DEVICE = "cuda:0"
# Absolute, so that a worker whose working directory a scheduler has changed
# (Ray moves each trial into its own directory) still finds the cached data.
DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")


# --- data ------------------------------------------------------------------


def _loaders(
    train: Any,
    valid: Any,
    batch_size: int,
    train_subset: int | None,
    valid_subset: int | None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    train_data = Subset(train, range(min(train_subset, len(train)))) if train_subset else train
    valid_data = Subset(valid, range(min(valid_subset, len(valid)))) if valid_subset else valid
    # In-process loading is the default; ``num_workers`` switches to persistent
    # worker processes, started once per loader rather than once per trial.
    persistent = num_workers > 0
    return (
        DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent,
        ),
        DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent,
        ),
    )


def load_fashion_mnist(
    batch_size, root=DATA_ROOT, train_subset=None, valid_subset=None, num_workers=0
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    train = FashionMNIST(root=root, train=True, download=True, transform=transform)
    valid = FashionMNIST(root=root, train=False, download=True, transform=transform)
    return _loaders(train, valid, batch_size, train_subset, valid_subset, num_workers)


def load_kmnist(batch_size, root=DATA_ROOT, train_subset=None, valid_subset=None, num_workers=0):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1918,), (0.3483,))]
    )
    train = KMNIST(root=root, train=True, download=True, transform=transform)
    valid = KMNIST(root=root, train=False, download=True, transform=transform)
    return _loaders(train, valid, batch_size, train_subset, valid_subset, num_workers)


def load_cifar10(batch_size, root=DATA_ROOT, train_subset=None, valid_subset=None, num_workers=0):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train = CIFAR10(root=root, train=True, download=True, transform=transform)
    valid = CIFAR10(root=root, train=False, download=True, transform=transform)
    return _loaders(train, valid, batch_size, train_subset, valid_subset, num_workers)


# --- models ----------------------------------------------------------------


class MLP(nn.Module):
    """Three linear layers with per-member dropout after each hidden layer.

    ``hidden`` sizes every weight and is therefore shape-changing.
    """

    def __init__(self, hidden: int = 256, dropout_default: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.dropout_default = dropout_default

    def _dropout(self, h: Tensor, hp: dict[str, Tensor] | None) -> Tensor:
        if hp is not None and "dropout" in hp:
            return member_dropout(h, p=hp["dropout"], training=self.training)
        if self.dropout_default > 0.0:
            p = torch.tensor(self.dropout_default, device=h.device)
            return member_dropout(h, p=p, training=self.training)
        return h

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        h = x.flatten(1)
        h = self._dropout(torch.relu(self.fc1(h)), hp)
        h = self._dropout(torch.relu(self.fc2(h)), hp)
        return self.fc3(h)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with attention written out explicitly.

    Attention is spelled out rather than taken from ``nn.MultiheadAttention``
    so that the per-member dropout can be applied inside it and so that every
    operation is one ``vmap`` already supports. LayerNorm carries parameters
    only, no running statistics, so it survives the vectorised call.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        # No bias on the key projection: a constant added to every key leaves
        # the softmax unchanged, so its gradient is zero in exact arithmetic and
        # pure rounding noise in float32, which Adam's normalised update
        # amplifies into steps that differ between implementations. The shared
        # qkv projection therefore has no bias, as in many ViTs.
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)

    def forward(self, x: Tensor, dropout: Callable[[Tensor], Tensor]) -> Tensor:
        batch, tokens, dim = x.shape
        head_dim = dim // self.heads
        qkv = self.qkv(self.norm1(x)).view(batch, tokens, 3, self.heads, head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [batch, tokens, heads, head_dim]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [batch, heads, tokens, head_dim]
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        attention = dropout(attention.softmax(dim=-1))
        mixed = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        x = x + dropout(self.proj(mixed))
        x = x + dropout(self.fc2(F.gelu(self.fc1(self.norm2(x)))))
        return x


class TinyViT(nn.Module):
    """A small Vision Transformer on 4x4 patches of a 32x32 image.

    ``embed_dim`` sizes every tensor and is shape-changing; depth and heads are
    fixed so that the search space has one architectural axis, as the other
    tasks do. Dropout is applied per member through ``hp``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        patch: int = 4,
        image: int = 32,
        channels: int = 3,
        classes: int = 10,
        dropout_default: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch = patch
        tokens = (image // patch) ** 2
        self.embed = nn.Linear(channels * patch * patch, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, tokens + 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, classes)
        self.dropout_default = dropout_default

    def _patchify(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        p = self.patch
        x = x.view(batch, channels, height // p, p, width // p, p)
        tokens = (height // p) * (width // p)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(batch, tokens, channels * p * p)

    def forward(self, x: Tensor, hp: dict[str, Tensor] | None = None) -> Tensor:
        if hp is not None and "dropout" in hp:
            p: Tensor | None = hp["dropout"]
        elif self.dropout_default > 0.0:
            p = torch.tensor(self.dropout_default, device=x.device)
        else:
            p = None

        def dropout(h: Tensor) -> Tensor:
            return h if p is None else member_dropout(h, p=p, training=self.training)

        h = self.embed(self._patchify(x))
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1) + self.pos
        h = dropout(h)
        for block in self.blocks:
            h = block(h, dropout)
        return self.head(self.norm(h[:, 0]))


INIT_SEED = 0
ORDER_SEED = 0


def _seeded_members(factory: Callable[[], nn.Module], k: int) -> list[nn.Module]:
    """``k`` members with identical initial weights, the plain objective's weights."""
    members = []
    for _ in range(k):
        torch.manual_seed(INIT_SEED)
        members.append(factory())
    return members


# --- task definitions --------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """Everything an arm needs to run one task.

    Attributes:
        name: Identifier used on the command line and in result files.
        load: ``load(batch_size, train_subset, valid_subset) -> (train, valid)``.
        vectorised: Names the vectorised objective declares in ``over``.
        shared: Names of the shape-changing hyperparameters.
        plain_objective: ``(train_loader, valid_loader, epochs) -> objective``
            for ``study.optimize``.
        vectorised_objective: The same for ``saturate``.
        parameter_range: Parameter counts across the architectural range.
    """

    name: str
    load: Callable[..., tuple[DataLoader, DataLoader]]
    vectorised: tuple[str, ...]
    shared: tuple[str, ...]
    plain_objective: Callable[[Any, Any, int], Callable[[optuna.Trial], float]]
    vectorised_objective: Callable[[Any, Any, int], Any]
    parameter_range: tuple[int, int]
    distributions: dict[str, Any]


def _plain_accuracy(model: nn.Module, loader: Any) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += int((model(x).argmax(dim=-1) == y).sum())
            total += int(y.shape[0])
    model.train()
    return correct / total


def _plain_train(model: nn.Module, optimiser: torch.optim.Optimizer, loader: Any, epochs: int):
    torch.manual_seed(ORDER_SEED)  # batch order and dropout masks start from the same state
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimiser.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimiser.step()


# fashion_cnn ---------------------------------------------------------------

FASHION_LR = (1e-3, 3e-1)
FASHION_DROPOUT = (0.0, 0.6)
FASHION_WIDTH = (8, 48)


def fashion_plain(train_loader, valid_loader, epochs):
    def objective(trial: optuna.Trial) -> float:
        width = trial.suggest_int("width", *FASHION_WIDTH)
        lr = trial.suggest_float("learning_rate", *FASHION_LR, log=True)
        dropout = trial.suggest_float("dropout", *FASHION_DROPOUT)
        torch.manual_seed(INIT_SEED)
        model = SmallCNN(width=width, dropout_default=dropout).to(DEVICE)
        _plain_train(model, torch.optim.SGD(model.parameters(), lr=lr), train_loader, epochs)
        return _plain_accuracy(model, valid_loader)

    return objective


def fashion_vectorised(train_loader, valid_loader, epochs):
    @osat.vectorizable(over=["learning_rate", "dropout"])
    def objective(trial: Any, ctx: Any) -> list[float]:
        width = trial.suggest_int("width", *FASHION_WIDTH)
        lr = trial.suggest_float("learning_rate", *FASHION_LR, log=True)
        dropout = trial.suggest_float("dropout", *FASHION_DROPOUT)
        ensemble = ctx.stack(_seeded_members(lambda: SmallCNN(width=width), ctx.k))
        optimiser = ctx.sgd(ensemble, lr=lr)
        torch.manual_seed(ORDER_SEED)
        for _ in range(epochs):
            for x, y in train_loader:
                ctx.step(ensemble, optimiser, x, y, member_hp={"dropout": dropout})
        return ctx.accuracy(ensemble, valid_loader)

    return objective


# kmnist_mlp ----------------------------------------------------------------

MLP_LR = (1e-3, 1e-1)
MLP_MOMENTUM = (0.0, 0.99)
MLP_DROPOUT = (0.0, 0.5)
MLP_WEIGHT_DECAY = (1e-6, 1e-3)
MLP_HIDDEN = (64, 512)


def mlp_plain(train_loader, valid_loader, epochs):
    def objective(trial: optuna.Trial) -> float:
        hidden = trial.suggest_int("hidden", *MLP_HIDDEN, log=True)
        lr = trial.suggest_float("learning_rate", *MLP_LR, log=True)
        momentum = trial.suggest_float("momentum", *MLP_MOMENTUM)
        dropout = trial.suggest_float("dropout", *MLP_DROPOUT)
        weight_decay = trial.suggest_float("weight_decay", *MLP_WEIGHT_DECAY, log=True)
        torch.manual_seed(INIT_SEED)
        model = MLP(hidden=hidden, dropout_default=dropout).to(DEVICE)
        optimiser = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        _plain_train(model, optimiser, train_loader, epochs)
        return _plain_accuracy(model, valid_loader)

    return objective


def mlp_vectorised(train_loader, valid_loader, epochs):
    @osat.vectorizable(over=["learning_rate", "momentum", "dropout", "weight_decay"])
    def objective(trial: Any, ctx: Any) -> list[float]:
        hidden = trial.suggest_int("hidden", *MLP_HIDDEN, log=True)
        lr = trial.suggest_float("learning_rate", *MLP_LR, log=True)
        momentum = trial.suggest_float("momentum", *MLP_MOMENTUM)
        dropout = trial.suggest_float("dropout", *MLP_DROPOUT)
        weight_decay = trial.suggest_float("weight_decay", *MLP_WEIGHT_DECAY, log=True)
        ensemble = ctx.stack(_seeded_members(lambda: MLP(hidden=hidden), ctx.k))
        optimiser = ctx.sgd(ensemble, lr=lr, momentum=momentum, weight_decay=weight_decay)
        torch.manual_seed(ORDER_SEED)
        for _ in range(epochs):
            for x, y in train_loader:
                ctx.step(ensemble, optimiser, x, y, member_hp={"dropout": dropout})
        return ctx.accuracy(ensemble, valid_loader)

    return objective


# cifar_vit -----------------------------------------------------------------

VIT_LR = (1e-4, 3e-3)
VIT_WEIGHT_DECAY = (1e-4, 1e-1)
VIT_DROPOUT = (0.0, 0.3)
VIT_EMBED = (32, 96)  # stepped by 16 so that every width divides into 4 heads


def vit_plain(train_loader, valid_loader, epochs):
    def objective(trial: optuna.Trial) -> float:
        embed_dim = trial.suggest_int("embed_dim", *VIT_EMBED, step=16)
        lr = trial.suggest_float("learning_rate", *VIT_LR, log=True)
        weight_decay = trial.suggest_float("weight_decay", *VIT_WEIGHT_DECAY, log=True)
        dropout = trial.suggest_float("dropout", *VIT_DROPOUT)
        torch.manual_seed(INIT_SEED)
        model = TinyViT(embed_dim=embed_dim, dropout_default=dropout).to(DEVICE)
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        _plain_train(model, optimiser, train_loader, epochs)
        return _plain_accuracy(model, valid_loader)

    return objective


def vit_vectorised(train_loader, valid_loader, epochs):
    @osat.vectorizable(over=["learning_rate", "weight_decay", "dropout"])
    def objective(trial: Any, ctx: Any) -> list[float]:
        embed_dim = trial.suggest_int("embed_dim", *VIT_EMBED, step=16)
        lr = trial.suggest_float("learning_rate", *VIT_LR, log=True)
        weight_decay = trial.suggest_float("weight_decay", *VIT_WEIGHT_DECAY, log=True)
        dropout = trial.suggest_float("dropout", *VIT_DROPOUT)
        ensemble = ctx.stack(_seeded_members(lambda: TinyViT(embed_dim=embed_dim), ctx.k))
        optimiser = ctx.adamw(ensemble, lr=lr, weight_decay=weight_decay)
        torch.manual_seed(ORDER_SEED)
        for _ in range(epochs):
            for x, y in train_loader:
                ctx.step(ensemble, optimiser, x, y, member_hp={"dropout": dropout})
        return ctx.accuracy(ensemble, valid_loader)

    return objective


def _count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


_D = optuna.distributions
FASHION_SPACE = {
    "width": _D.IntDistribution(*FASHION_WIDTH),
    "learning_rate": _D.FloatDistribution(*FASHION_LR, log=True),
    "dropout": _D.FloatDistribution(*FASHION_DROPOUT),
}
MLP_SPACE = {
    "hidden": _D.IntDistribution(*MLP_HIDDEN, log=True),
    "learning_rate": _D.FloatDistribution(*MLP_LR, log=True),
    "momentum": _D.FloatDistribution(*MLP_MOMENTUM),
    "dropout": _D.FloatDistribution(*MLP_DROPOUT),
    "weight_decay": _D.FloatDistribution(*MLP_WEIGHT_DECAY, log=True),
}
VIT_SPACE = {
    "embed_dim": _D.IntDistribution(*VIT_EMBED, step=16),
    "learning_rate": _D.FloatDistribution(*VIT_LR, log=True),
    "weight_decay": _D.FloatDistribution(*VIT_WEIGHT_DECAY, log=True),
    "dropout": _D.FloatDistribution(*VIT_DROPOUT),
}


TASKS: dict[str, Task] = {
    "fashion_cnn": Task(
        name="fashion_cnn",
        load=load_fashion_mnist,
        vectorised=("learning_rate", "dropout"),
        shared=("width",),
        plain_objective=fashion_plain,
        vectorised_objective=fashion_vectorised,
        parameter_range=(_count(SmallCNN(width=8)), _count(SmallCNN(width=48))),
        distributions=FASHION_SPACE,
    ),
    "kmnist_mlp": Task(
        name="kmnist_mlp",
        load=load_kmnist,
        vectorised=("learning_rate", "momentum", "dropout", "weight_decay"),
        shared=("hidden",),
        plain_objective=mlp_plain,
        vectorised_objective=mlp_vectorised,
        parameter_range=(_count(MLP(hidden=64)), _count(MLP(hidden=512))),
        distributions=MLP_SPACE,
    ),
    "cifar_vit": Task(
        name="cifar_vit",
        load=load_cifar10,
        vectorised=("learning_rate", "weight_decay", "dropout"),
        shared=("embed_dim",),
        plain_objective=vit_plain,
        vectorised_objective=vit_vectorised,
        parameter_range=(_count(TinyViT(embed_dim=32)), _count(TinyViT(embed_dim=96))),
        distributions=VIT_SPACE,
    ),
}
