import logging
import sys
import time

sys.path.append("../")

import gym
import torch
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import QREPS
from qreps.algorithms.sampler import BestResponseSampler, ExponentiatedGradientSampler
from qreps.feature_functions import (
    FeatureConcatenation,
    IdentityFeature,
    NNFeatures,
    OneHotFeature,
)
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.utilities.util import set_seed
from qreps.valuefunctions import NNQFunction, SimpleQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SEED = 1234
set_seed(SEED)

qreps_config = {
    "eta": 0.45,
    "beta": 2e-2,
    "saddle_point_steps": 300,
    "policy_opt_steps": 450,
    "policy_lr": 2e-5,
    "discount": 0.99,
}

timestamp = time.time()
gym_env = gym.make("CartPole-v0")
gym_env.seed(SEED)
# gym_env = gym.wrappers.Monitor(gym_env, directory=f"./frozen_lake_{timestamp}", video_callable=lambda x: True)

env = gym_wrapper.DMEnvFromGym(gym_env)
num_obs = env.observation_spec().shape[0]
num_act = env.action_spec().num_values
print(env.observation_spec())


def train(config: dict):

    q_function = NNQFunction(
        obs_dim=num_obs, act_dim=num_act, feature_fn=IdentityFeature()
    )
    policy = CategoricalMLP(num_obs, 2)

    writer = SummaryWriter()

    agent = QREPS(
        writer=writer,
        policy=policy,
        q_function=q_function,
        learner=torch.optim.Adam,
        sampler=BestResponseSampler,
        reward_transformer=lambda r: r / 1000,
        **config,
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=30, max_steps=200, number_rollouts=5)


wandb.init(
    project="qreps",
    entity="sebimarkgraf",
    sync_tensorboard=True,
    tags=["cartpole_hyperparam", "qreps"],
    job_type="hyperparam",
    config=qreps_config,
    monitor_gym=True,
)
train(wandb.config)
wandb.finish()
