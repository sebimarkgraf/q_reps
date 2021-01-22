import torch


def to_torch(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.get_default_dtype())


def center_advantages(advantages, eps=1e-12):
    return (advantages - torch.mean(advantages)) / (torch.std(advantages) + eps)


def positive_advantages(advantages, eps=1e-12):
    return advantages - torch.min(advantages)
