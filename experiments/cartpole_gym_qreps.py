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
from qreps.algorithms.sampler import BestResponseSampler, ExponentitedGradientSampler
from qreps.feature_functions import FeatureConcatenation, NNFeatures, OneHotFeature
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.utilities.util import set_seed
from qreps.valuefunctions import SimpleQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

SEED = 1234
set_seed(SEED)

qreps_config = {
    "eta": 1.91,
    "beta": 0.02766,
    "saddle_point_steps": 300,
    "policy_opt_steps": 450,
    "policy_lr": 0.00002,
    "discount": 0.99,
    "average_weights": True,
    "grad_samples": 5,
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
    obs_feature_fn = NNFeatures(num_obs, feat_dim=200)

    q_function = SimpleQFunction(
        obs_dim=200, act_dim=num_act, feature_fn=obs_feature_fn
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
