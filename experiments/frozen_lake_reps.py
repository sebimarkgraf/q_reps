import sys
import time

import gym

sys.path.append("../")

import logging

from bsuite.utils import gym_wrapper
from gym.envs.toy_text import FrozenLakeEnv
from gym_minigrid.wrappers import *
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import QREPS, REPS
from qreps.feature_functions import FeatureConcatenation, OneHotFeature
from qreps.policies import StochasticTablePolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import SimpleQFunction, SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


timestamp = time.time()
gym_env = gym.make("FrozenLake-v0")
gym_env = gym.wrappers.Monitor(
    gym_env, directory=f"./frozen_lake_{timestamp}", video_callable=lambda x: True
)
env = gym_wrapper.DMEnvFromGym(gym_env)
print(env.observation_spec())
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values


config = {
    "discount": 0.9,
    "eta": 2.0,
    "dual_lr": 0.07,
    "policy_lr": 0.005,
    "entropy_constrained": True,
    "dual_opt_steps": 300,
    "policy_opt_steps": 300,
}


def train(config: dict):
    feature_fn = OneHotFeature(obs_num)

    value_function = SimpleValueFunction(obs_dim=obs_num, feature_fn=feature_fn,)

    policy = StochasticTablePolicy(obs_num, act_num)

    writer = SummaryWriter()

    agent = REPS(writer=writer, policy=policy, value_function=value_function, **config)

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=40, max_steps=300, number_rollouts=5)

    print("Policy", policy._policy)


wandb.init(
    project="qreps",
    entity="sebimarkgraf",
    sync_tensorboard=True,
    tags=["frozen_lake", "qreps "],
    job_type="hyperparam",
    config=config,
)
train(wandb.config)
wandb.finish()
