import logging
import sys
import time

sys.path.append("../")

import gym
import numpy as np
import torch
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import REPS
from qreps.feature_functions import NNFeatures
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

reps_config = {
    "discount": 0.99,
    "eta": 0.001,
    "dual_lr": 2e-2,
    "policy_lr": 2e-5,
    "entropy_constrained": False,
    "dual_opt_steps": 300,
    "policy_opt_steps": 300,
}

timestamp = time.time()
gym_env = gym.make("MountainCar-v0")
gym_env.seed(SEED)
gym_env = gym.wrappers.Monitor(
    gym_env, directory=f"./mount_car_{timestamp}", video_callable=lambda x: True
)
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
    trainer.train(num_iterations=30, max_steps=200, number_rollouts=20)


wandb.init(
    project="qreps",
    entity="sebimarkgraf",
    sync_tensorboard=True,
    tags=["mountain_car", "reps"],
    job_type="hyperparam",
    config=reps_config,
)
train(wandb.config)
wandb.finish()
