import logging

import torch
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_control.suite.cartpole import balance
from ray import tune
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import REPS
from qreps.policies import GaussianMLPStochasticPolicy
from qreps.utilities.observation_transform import OrderedDictFlattenTransform
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import NNValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

env: Environment = balance(
    time_limit=10.0, environment_kwargs={"flat_observation": True}
)

config = {
    "sigma": 0.1,
    "num_rollouts": 10,
    "gamma": 1.0,
    "eta": 1e-5,
    "dual_lr": 2e-3,
    "lr": 2e-3,
    "max_steps": 500,
    "pol_opt_steps": 600,
    "dual_opt_steps": 150,
}
writer = SummaryWriter(comment="_mujuco_reps_optimized")


def train(config: dict):
    policy = GaussianMLPStochasticPolicy(5, 1, sigma=config["sigma"])

    value_function = NNValueFunction(obs_dim=5)

    agent = REPS(
        policy=policy,
        value_function=value_function,
        entropy_constrained=False,
        discount=config["gamma"],
        eta=config["eta"],
        dual_lr=config["dual_lr"],
        policy_lr=config["lr"],
        writer=writer,
        dual_opt_steps=config["dual_opt_steps"],
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(
        num_iterations=100,
        max_steps=config["max_steps"],
        number_rollouts=config["num_rollouts"],
    )
    policy.set_eval_mode(True)
    return agent


agent = train(config)


def eval_func(timestep):
    action = agent.select_action(timestep)
    return action


viewer.launch(env, eval_func)
