import torch


def center_advantages(advantages, eps=1e-12):
    return (advantages - torch.mean(advantages)) / (torch.std(advantages) + eps)


def positive_advantages(advantages):
    min = torch.min(advantages)
    return advantages - min
