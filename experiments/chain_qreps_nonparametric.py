import sys

from qreps.algorithms.saddle_qreps import SaddleQREPS
from qreps.policies.qreps_policy import QREPSPolicy
from qreps.valuefunctions.AccumulativeModule import AccumulativeModule

sys.path.append("../")

import logging

import torch
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import QREPS
from qreps.feature_functions import OneHotFeature
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import SimpleQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


gym_env = NChainEnv(n=5)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values


qreps_config = {
    "eta": 1.0,
    "beta": 0.05,
    "saddle_point_steps": 300,
    "policy_opt_steps": 300,
    "discount": 0.99,
    "optimize_policy": False,
}


def train(config: dict):
    feature_fn = OneHotFeature(obs_num)

    value_function = AccumulativeModule(
        SimpleQFunction(obs_dim=obs_num, act_dim=act_num, feature_fn=feature_fn,)
    )

    temp = config["eta"]
    policy = QREPSPolicy(q_function=value_function, temp=temp)

    writer = SummaryWriter()

    agent = SaddleQREPS(
        writer=writer,
        policy=policy,
        q_function=value_function,
        learner=torch.optim.SGD,
        **config
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=30, max_steps=200, number_rollouts=1)


wandb.init(
    project="qreps",
    entity="sebimarkgraf",
    sync_tensorboard=True,
    tags=["chain", "qreps_nonparametric"],
    job_type="experiment",
    config=qreps_config,
)
train(wandb.config)
wandb.finish()
