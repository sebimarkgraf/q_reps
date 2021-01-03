import logging

import gym
import torch
from bsuite.baselines.experiment import run
from bsuite.utils import gym_wrapper

from qreps.policy import DiscreteStochasticPolicy
from qreps.reps import REPS

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

env = gym_wrapper.DMEnvFromGym(gym.make("FrozenLake-v0"))

obs_num = env.observation_spec().num_values


def feature_fn(x):
    return torch.nn.functional.one_hot(torch.tensor(x, dtype=torch.int64), obs_num).to(
        torch.float32
    )


agent = REPS(
    feat_shape=(obs_num,),
    sequence_length=100,
    feature_fn=feature_fn,
    epsilon=1e-5,
    policy=DiscreteStochasticPolicy(
        env.observation_spec().num_values, env.action_spec().num_values
    ),
)


run(agent, env, num_episodes=100000)
