from typing import Callable

import torch


def to_torch(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x).float()


def center_advantages(advantages, eps=1e-12):
    return (advantages - torch.mean(advantages)) / (torch.std(advantages) + eps)


def positive_advantages(advantages, eps=1e-12):
    return advantages - torch.min(advantages)


def integrate_discrete(func, distribution: torch.distributions.Distribution):
    value = 0
    for action in distribution.enumerate_support():
        q_values = func(action)
        log_probs = distribution.log_prob(action)
        value += q_values * torch.exp(log_probs.detach())
    return value


def integrate_continuous(
    func: Callable, distribution: torch.distributions.Distribution, samples=15
):
    value = torch.zeros(distribution.batch_shape)
    for _ in range(samples):
        if distribution.has_rsample:
            action = distribution.rsample().detach()
        else:
            action = distribution.sample().detach()
        value += func(action)
    return value / samples


def integrate(
    func: Callable,
    distribution: torch.distributions.Distribution,
    continuous_samples=15,
):
    if distribution.has_enumerate_support:
        return integrate_discrete(func, distribution)
    else:
        return integrate_continuous(func, distribution, continuous_samples)


def episode_normalize_rewards(rewards):
    return (rewards - rewards.mean(0)) / rewards.std()
