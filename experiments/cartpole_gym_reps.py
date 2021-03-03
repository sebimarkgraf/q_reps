import logging
import time

import gym
import torch
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import REPS
from qreps.feature_functions import NNFeatures
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import NNValueFunction, SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


torch.manual_seed(1234)

reps_config = {
    "discount": 1.0,
    "eta": 5.0,
    "dual_lr": 0.01,
    "policy_lr": 5e-4,
    "entropy_constrained": False,
    "dual_opt_steps": 300,
}

timestamp = time.time()
gym_env = gym.make("CartPole-v0")
gym_env = gym.wrappers.Monitor(gym_env, directory=f"./frozen_lake_{timestamp}")
env = gym_wrapper.DMEnvFromGym(gym_env)

num_obs = env.observation_spec().shape[0]


def train(config: dict):
    feature_fn = NNFeatures(num_obs, feat_dim=200)
    value_function = SimpleValueFunction(obs_dim=200, feature_fn=feature_fn)
    policy = CategoricalMLP(num_obs, 2)
    writer = SummaryWriter()

    agent = REPS(writer=writer, policy=policy, value_function=value_function, **config)

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=30, max_steps=200, number_rollouts=5)


wandb.init(
    project="qreps",
    entity="sebimarkgraf",
    sync_tensorboard=True,
    tags=["cartpole_hyperparam", "reps"],
    job_type="hyperparam",
    config=reps_config,
)
train(wandb.config)
wandb.finish()
