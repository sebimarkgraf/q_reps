import torch
import torch.nn.functional as F
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import FrozenLakeEnv
from ray import tune
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.reps import REPS
from qreps.policies.stochastic_table import StochasticTablePolicy
from qreps.trainer import Trainer
from qreps.valuefunctions.value_functions import NNValueFunction, SimpleValueFunction

gym_env = FrozenLakeEnv(is_slippery=False)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

config = {
    "num_rollouts": 20,
    "gamma": 0.9,
    "eta": 0.1,
    "dual_lr": 2e-4,
    "lr": 2e-4,
}

writer = SummaryWriter(comment="_frozen_lake_reps")


def train(config: dict):
    def feature_fn(x):
        return F.one_hot(x.long(), obs_num).float()

    value_function = NNValueFunction(obs_dim=obs_num, feature_fn=feature_fn)

    policy = StochasticTablePolicy(obs_num, act_num)

    agent = REPS(
        buffer_size=5000,
        policy=policy,
        value_function=value_function,
        gamma=config["gamma"],
        eta=config["eta"],
        dual_lr=config["dual_lr"],
        lr=config["lr"],
        writer=writer,
        pol_opt_steps=300,
    )

    trainer = Trainer()
    trainer.setup(agent, env)

    for i in range(10):
        trainer.train(
            num_iterations=10, max_steps=30, number_rollouts=config["num_rollouts"]
        )
        policy.set_eval_mode(True)
        val_reward = trainer.validate(5, 30)
        policy.set_eval_mode(False)
        tune.report(reward=torch.sum(torch.tensor(val_reward)).item())

        timestep = env.reset()
        while not timestep.last():
            # Generate an action from the agent's policy.
            action = agent.select_action(timestep)
            # Step the environment.
            new_timestep = env.step(action)
            # Book-keeping.
            timestep = new_timestep
            gym_env.render()


train(config)
