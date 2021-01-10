import logging

import torch
from bsuite.baselines.experiment import run
from dm_control import suite, viewer
from dm_control.rl.control import Environment

from qreps.fourier_features import FourierFeatures
from qreps.observation_transform import OrderedDictFlattenTransform
from qreps.policy import GaussianMLP
from qreps.reps import REPS

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

env: Environment = suite.load(domain_name="cartpole", task_name="balance")


feature_fn = FourierFeatures(5, 75)
policy = GaussianMLP(75, 1, minimizing_epochs=300)

agent = OrderedDictFlattenTransform(
    REPS(
        feat_shape=(75,),
        sequence_length=2000,
        val_feature_fn=feature_fn,
        pol_feature_fn=feature_fn,
        epsilon=1e-5,
        policy=policy,
    )
)

run(agent, env, num_episodes=200)

policy.set_eval_mode(True)


def eval_func(timestep):
    action = agent.select_action(timestep)
    return action


viewer.launch(env, eval_func)
