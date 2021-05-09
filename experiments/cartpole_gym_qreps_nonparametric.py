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
from qreps.algorithms.sampler import BestResponseSampler
from qreps.feature_functions import IdentityFeature
from qreps.policies.qreps_policy import QREPSPolicy
from qreps.utilities.trainer import Trainer
from qreps.utilities.util import set_seed
from qreps.valuefunctions import NNQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SEED = 24
set_seed(SEED)

qreps_config = {
    "eta": 0.1,
    "beta": 2e-2,
    "saddle_point_steps": 300,
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
    policy = QREPSPolicy(q_function=q_function, temp=config["eta"])

    writer = SummaryWriter()

    agent = QREPS(
        writer=writer,
        policy=policy,
        q_function=q_function,
        learner=torch.optim.Adam,
        sampler=BestResponseSampler,
        optimize_policy=False,
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
    tags=["cartpole", "qreps_nonparametric"],
    job_type="hyperparam",
    config=qreps_config,
    monitor_gym=True,
)
train(wandb.config)
wandb.finish()
