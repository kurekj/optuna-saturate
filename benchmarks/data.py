"""Fashion-MNIST for the evaluation scripts.

Downloaded through torchvision rather than parsed by hand: the IDX format has
header offsets and a byte order that are easy to get subtly wrong, and getting
them wrong would corrupt every number the benchmark produces.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# Channel statistics of the Fashion-MNIST training split.
MEAN = 0.2860
STD = 0.3530


def load_fashion_mnist(
    batch_size: int = 128,
    root: str = "data",
    train_subset: int | None = None,
    valid_subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Training and validation loaders, downloading the dataset on first use.

    Args:
        batch_size: Batch size for both loaders.
        root: Cache directory. Ignored by git.
        train_subset: Keep only the first N training examples. Shortens a sweep
            without changing anything else about it.
        valid_subset: Keep only the first N validation examples.

    Returns:
        ``(train_loader, valid_loader)``. Shuffling uses PyTorch's global seed, so
        seeding before the call makes the order reproducible.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train = FashionMNIST(root=root, train=True, download=True, transform=transform)
    valid = FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_data = Subset(train, range(min(train_subset, len(train)))) if train_subset else train
    valid_data = Subset(valid, range(min(valid_subset, len(valid)))) if valid_subset else valid

    # num_workers=0 on purpose: worker processes would each need their own CUDA
    # context, and the loader is not the bottleneck at this model size.
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0),
    )
