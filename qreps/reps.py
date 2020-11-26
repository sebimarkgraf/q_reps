from typing import Any, Callable, NamedTuple, Tuple

import dm_env
import numpy as np
import torch
from bsuite.baselines import base
from bsuite.baselines.utils import sequence
from dm_env import specs


class REPS(base.Agent):
    """Feed-forward actor-critic agent."""

    _buffer: sequence.Buffer

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        optimizer: torch.Optimizer,
        sequence_length: int,
        eta,
        epsilon,
        discount: float,
    ):
        self.eta = eta
        self.epsilon = epsilon
        self.action_spec = action_spec
        self.obs_spec = obs_spec
        self.buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)

    def _dual_compute(self, eps, eta, theta, features, rewards):
        F_mean = features.mean(0).view(1, -1)
        R_over_eta = (rewards - features.mm(theta)) / eta
        R_over_eta_max = R_over_eta.max()
        Z = torch.exp(R_over_eta - R_over_eta_max)
        Z_sum = Z.sum()
        log_sum_exp = R_over_eta_max + torch.log(Z_sum / features.shape[0])

        f = eta * (self.epsilon + log_sum_exp) + F_mean.mm(theta)

        d_eta = eps + log_sum_exp - Z.t().mm(R_over_eta) / Z_sum
        d_theta = F_mean - (Z.t().mm(features) / Z_sum)
        return f.numpy(), np.append(d_eta.numpy(), d_theta.numpy())

    def _dual(self, eta, epsilon, bellmann_error):
        return (
            eta
            * epsilon
            * torch.log(torch.mean(torch.exp(epsilon + 1 / eta * bellmann_error)))
        )

    def _bellmann_error(self, theta, features, features_next, rewards):
        return torch.mean(rewards + theta.mm(features_next) - theta.mm(features))

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions using current policy in an on-policy setting"""
        ## TODO: Implement action selection
        action = self.action_spec.generate_value()
        return int(action)

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: base.Action,
        new_timestep: dm_env.TimeStep,
    ):
        """Sampling: Obtain N samples (s_i, a_i, s_i', r_i)

        Relevant from experiment
        for _ in range(num_episodes):
        # Run an episode.
        timestep = environment.reset()
        while not timestep.last():
          # Generate an action from the agent's policy.
          action = agent.select_action(timestep)

          # Step the environment.
          new_timestep = environment.step(action)

          # Tell the agent about what just happened.
          agent.update(timestep, action, new_timestep)

          # Book-keeping.
          timestep = new_timestep

        """
        self._buffer.append(timestep, action, new_timestep)
        if self._buffer.full() or new_timestep.last():
            trajectory = self._buffer.drain()
            self._state = self._sgd_step(self._state, trajectory)
