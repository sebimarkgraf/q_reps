import sys

sys.path.append("../")

import torch
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from tensorboardX import SummaryWriter

import wandb
from qreps.algorithms import REPS
from qreps.feature_functions import OneHotFeature
from qreps.policies import StochasticTablePolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import SimpleValueFunction

torch.manual_seed(1234)

gym_env = NChainEnv(n=5)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

config = {
    "discount": 0.99,
    "eta": 2.0,
    "dual_lr": 0.07,
    "policy_lr": 0.005,
    "entropy_constrained": True,
    "dual_opt_steps": 300,
    "policy_opt_steps": 300,
}


def train(config: dict):
    value_function = SimpleValueFunction(
        obs_dim=obs_num, feature_fn=OneHotFeature(obs_num)
    )
    policy = StochasticTablePolicy(obs_num, act_num)

    writer = SummaryWriter()

    agent = REPS(policy=policy, value_function=value_function, writer=writer, **config)

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=10, max_steps=200, number_rollouts=1)


wandb.init(
    project="qreps",
    entity="sebimarkgraf",
    sync_tensorboard=True,
    tags=["chain_profiling"],
    job_type="hyperparam",
    config=config,
)
train(wandb.config)
wandb.finish()
