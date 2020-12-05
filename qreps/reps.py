import dm_env
import nlopt
import numpy as np
import torch
from bsuite.baselines import base
from bsuite.baselines.utils import sequence
from dm_env import specs

from .reps_dual import reps_dual


class REPS(base.Agent):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        sequence_length: int,
        epsilon=1e-5,
        dual_optimizer_algorithm=nlopt.LD_LBFGS,
    ):
        print("Observations:", obs_spec)
        print("Actions:", action_spec)
        self.eta = torch.rand((1,))
        self.epsilon = torch.tensor(epsilon)
        self.action_spec = action_spec
        self.obs_spec = obs_spec
        self.buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
        self.policy = torch.ones((action_spec.num_values, np.prod(obs_spec.shape)))
        self.theta = torch.rand(np.prod(obs_spec.shape)) + 1 / 2
        self.dual_optimizer_algorithm = dual_optimizer_algorithm

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions using current policy in an on-policy setting"""
        obs = torch.tensor(timestep.observation).flatten()
        m = torch.distributions.Categorical(self.policy.mv(obs))
        action = m.sample()
        return int(action)

    def value_function(self, features):
        return self.theta.dot(features)

    def optimize_dual(self, trajectory: sequence.Trajectory):
        observations, actions, rewards, _ = trajectory
        features = torch.tensor(observations[:-1]).flatten(start_dim=1)
        features_next = torch.tensor(observations[1:]).flatten(start_dim=1)
        rewards = torch.tensor(rewards)

        def eval_fn(x: np.array, grad: np.array):
            # TODO: Switch to analytical gradients
            eta = torch.tensor(
                x[0], dtype=torch.get_default_dtype(), requires_grad=True
            )
            theta = torch.tensor(
                x[1:], dtype=torch.get_default_dtype(), requires_grad=True
            )
            dual_val, d_eta, d_theta = reps_dual(
                eta, theta, features, features_next, rewards, self.epsilon
            )
            dual_val.backward()

            if grad.size > 0:
                grad[0] = eta.grad.numpy()
                grad[1:] = theta.grad.numpy()

            print("Dual_loss:", dual_val.item())
            return dual_val.item()

        optimizer = nlopt.opt(self.dual_optimizer_algorithm, 1 + len(self.theta))
        optimizer.set_lower_bounds([1e-2] + [-np.inf] * self.theta.shape[0])
        optimizer.set_min_objective(eval_fn)
        optimizer.set_ftol_abs(1e-2)

        x0 = [1] + [1] * self.theta.shape[0]
        params_best = optimizer.optimize(x0)
        self.eta = torch.tensor(params_best[0])
        self.theta = torch.from_numpy(params_best[1:]).view(-1, 1)

    def update_policy(self, trajectory: sequence.Trajectory):
        self.optimize_dual(trajectory)
        # Determine Value function?

        # Compute new policy

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
        self.buffer.append(timestep, action, new_timestep)
        if self.buffer.full() or new_timestep.last():
            trajectory = self.buffer.drain()
            self.update_policy(trajectory)
