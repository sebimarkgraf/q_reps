import dm_env
import nlopt
import numpy as np
import torch
from bsuite.baselines import base
from bsuite.baselines.utils import sequence
from dm_env import specs

from .feature_functions import bellman_error, bellman_error_batched
from .reps_dual import reps_dual


class REPS(base.Agent):
    """Feed-forward actor-critic agent."""

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        sequence_length: int,
        epsilon=1e-5,
        dual_optimizer_algorithm=nlopt.LD_SLSQP,
        policy=None,
    ):
        print("Observations:", obs_spec)
        print("Actions:", action_spec)
        self.eta = torch.rand((1,))
        self.epsilon = torch.tensor(epsilon)
        self.action_spec = action_spec
        self.obs_spec = obs_spec
        obs_shape = obs_spec.shape
        if obs_shape == ():
            obs_shape = (1,)
        self.buffer = sequence.Buffer(obs_spec, action_spec, sequence_length)
        self.theta = torch.rand(obs_shape)
        self.dual_optimizer_algorithm = dual_optimizer_algorithm
        self.policy = policy

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        """Selects actions using current policy in an on-policy setting"""
        # obs = torch.tensor(timestep.observation).flatten()
        # m = torch.distributions.Categorical(self.policy.mv(obs))
        observation = torch.tensor(timestep.observation).flatten()
        action = self.policy.sample(observation)
        return int(action)

    def value_function(self, features):
        return self.theta.dot(features)

    def optimize_dual(self, trajectory: sequence.Trajectory):
        observations, actions, rewards, _ = trajectory
        features = torch.tensor(
            observations[:-1], dtype=torch.get_default_dtype()
        ).unsqueeze(
            1
        )  # .flatten(start_dim=1)
        features_next = torch.tensor(
            observations[1:], dtype=torch.get_default_dtype()
        ).unsqueeze(
            1
        )  # .flatten(start_dim=1)
        rewards = torch.tensor(rewards, dtype=torch.get_default_dtype())

        def eval_fn(x: np.array, grad: np.array):
            eta = torch.tensor(x[0], dtype=torch.get_default_dtype())
            theta = torch.tensor(x[1:], dtype=torch.get_default_dtype())
            dual_val, d_eta, d_theta = reps_dual(
                eta, theta, features, features_next, rewards, self.epsilon
            )

            if grad.size > 0:
                grad[0] = d_eta.numpy()
                grad[1:] = d_theta.numpy()

            return dual_val.item()

        # SLSQP seems to work but L-BFGS fails without useful errors
        optimizer = nlopt.opt(self.dual_optimizer_algorithm, 1 + len(self.theta))
        optimizer.set_lower_bounds([1e-2] + [-np.inf] * len(self.theta))
        optimizer.set_min_objective(eval_fn)
        optimizer.set_ftol_abs(1e-3)
        optimizer.set_maxeval(3000)

        x0 = [1] + [1] * self.theta.shape[0]
        params_best = optimizer.optimize(x0)
        print("New Params:", params_best)
        print("Optimized dual value:", optimizer.last_optimum_value())
        if optimizer.last_optimize_result() == nlopt.MAXEVAL_REACHED:
            print("Stopped due to max eval iterations.")

        # Update params from optimum
        self.eta = torch.tensor(params_best[0])
        self.theta = torch.tensor(params_best[1:], dtype=torch.get_default_dtype())

    def update_policy(self, trajectory: sequence.Trajectory):
        self.optimize_dual(trajectory)

        observations, actions, rewards, _ = trajectory
        features = torch.tensor(
            observations[:-1], dtype=torch.get_default_dtype()
        ).unsqueeze(
            1
        )  # .flatten(start_dim=1)
        features_next = torch.tensor(
            observations[1:], dtype=torch.get_default_dtype()
        ).unsqueeze(
            1
        )  # .flatten(start_dim=1)
        rewards = torch.tensor(rewards, dtype=torch.get_default_dtype())

        # Calculate weights
        print(self.theta, features.dtype, features_next.dtype, rewards.dtype)
        weights = torch.exp(
            1
            / self.eta
            * bellman_error_batched(self.theta, features, features_next, rewards)
        )

        self.policy.fit(trajectory, weights)

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
