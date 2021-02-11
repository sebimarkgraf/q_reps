import torch
import torch.nn.functional as F
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import FrozenLakeEnv
from ray import tune
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.reps import REPS
from qreps.policies.stochastic_table import StochasticTablePolicy
from qreps.trainer import Trainer
from qreps.valuefunctions.value_functions import SimpleValueFunction

gym_env = FrozenLakeEnv()
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

config = {
    "num_rollouts": 5,
    "gamma": 1.0,
    "eta": 0.1,
    "dual_lr": 2e-2,
    "lr": 2e-2,
}

writer = SummaryWriter(comment="_frozen_lake_reps")


def train(config: dict):
    def feature_fn(x):
        return F.one_hot(x.long(), obs_num).float()

    value_function = SimpleValueFunction(obs_dim=obs_num, feature_fn=feature_fn)

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
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(
        num_iterations=50, max_steps=30, number_rollouts=config["num_rollouts"]
    )
    policy.set_eval_mode(True)
    val_reward = trainer.validate(5, 100)
    tune.report(reward=torch.sum(torch.tensor(val_reward)).item())


train(config)
