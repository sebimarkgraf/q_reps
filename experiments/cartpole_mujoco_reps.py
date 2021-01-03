import logging

import torch
from bsuite.baselines.experiment import run
from dm_control import suite

from qreps.observation_transform import OrderedDictFlattenTransform
from qreps.policy import GaussianMLP
from qreps.reps import REPS

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


env = suite.load(domain_name="cartpole", task_name="swingup")


def feature_fn(x):
    return torch.tensor(x, dtype=torch.get_default_dtype())


agent = OrderedDictFlattenTransform(
    REPS(
        feat_shape=(5,),
        sequence_length=50,
        feature_fn=feature_fn,
        epsilon=1e-5,
        policy=GaussianMLP((), ()),
    )
)


run(agent, env, num_episodes=5)
