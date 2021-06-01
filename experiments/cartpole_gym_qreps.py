import logging
import sys

from qreps.utilities.util import set_seed

sys.path.append("../")

import gym
import ray.tune as tune
import torch
from bsuite.utils import gym_wrapper
from ray.tune.suggest import Repeater
from ray.tune.suggest.hebo import HEBOSearch
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import QREPS
from qreps.feature_functions import IdentityFeature
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import NNQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 0

config = {
    "eta": tune.loguniform(2e-3, 2e-1),
    "alpha": tune.loguniform(2e-3, 2e-1),
    "beta": tune.loguniform(2e-4, 2e-1),
    "saddle_point_steps": tune.choice([300, 450]),
    "policy_opt_steps": tune.choice([300, 450]),
    "policy_lr": tune.loguniform(2e-3, 2e-1),
    "discount": 0.99,
}


def train(config: dict):
    seed = config["__trial_index__"] + SEED_OFFSET
    gym_env = gym.make("CartPole-v0")
    gym_env.seed(seed)
    set_seed(seed)
    env = gym_wrapper.DMEnvFromGym(gym_env)
    num_obs = env.observation_spec().shape[0]
    num_act = env.action_spec().num_values

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
        reward_transformer=lambda r: r / 1000,
        **config,
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(
        num_iterations=30,
        max_steps=200,
        number_rollouts=5,
        logging_callback=lambda r: tune.report(reward=r),
    )


search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=5)

# Repeat 2 samples 10 times each.
analysis = tune.run(train, num_samples=5, config=config, search_alg=re_search_alg)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df

df.to_csv("qreps_analysis.csv")
