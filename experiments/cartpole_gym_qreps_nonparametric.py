import argparse
import logging
import sys

from ray import tune
from ray.tune.suggest import Repeater
from ray.tune.suggest.hebo import HEBOSearch

sys.path.append("../")


import gym
import torch
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import QREPS
from qreps.algorithms.sampler import BestResponseSampler
from qreps.feature_functions import NNFeatures
from qreps.policies.qreps_policy import QREPSPolicy
from qreps.utilities.trainer import Trainer
from qreps.utilities.util import set_seed
from qreps.valuefunctions import SimpleQFunction

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
    "discount": 0.99,
}


def train(config: dict):
    seed = config["__trial_index__"] + SEED_OFFSET
    set_seed(seed)
    gym_env = gym.make("CartPole-v0")
    gym_env.seed(seed)
    # gym_env = gym.wrappers.Monitor(gym_env, directory=f"./frozen_lake_{timestamp}", video_callable=lambda x: True)

    env = gym_wrapper.DMEnvFromGym(gym_env)
    num_obs = env.observation_spec().shape[0]
    num_act = env.action_spec().num_values
    FEAT_DIM = 200
    feature_fn = NNFeatures(num_obs, feat_dim=FEAT_DIM)
    q_function = SimpleQFunction(
        obs_dim=FEAT_DIM, act_dim=num_act, feature_fn=feature_fn
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
    trainer.train(
        num_iterations=30,
        max_steps=200,
        number_rollouts=5,
        logging_callback=lambda r: tune.report(reward=r),
    )


search_alg = HEBOSearch(metric="reward", mode="max")
re_search_alg = Repeater(search_alg, repeat=5)

# Repeat 2 samples 10 times each.
analysis = tune.run(
    train,
    num_samples=5,
    config=config,
    search_alg=re_search_alg,
    local_dir="/home/temp_store/seb_markgraf/qreps_results",
)

print("Best config: ", analysis.get_best_config(metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df


df.to_csv("qreps_non_parametric_analysis.csv")
