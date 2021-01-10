import torch


def to_torch(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.get_default_dtype())
