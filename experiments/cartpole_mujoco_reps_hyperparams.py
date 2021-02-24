import logging

import torch
from dm_control.rl.control import Environment
from dm_control.suite.cartpole import balance
from ray import tune

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
    "sigma": 1.0,
    "num_rollouts": tune.grid_search([3, 5, 10, 15]),
    "gamma": tune.uniform(0.8, 1.0),
    "eta": tune.loguniform(2e-4, 2e-1, 10),
    "dual_lr": tune.loguniform(1e-3, 1e-1),
    "lr": tune.loguniform(1e-3, 1e-1),
    "max_steps": 200,
}


def train(config: dict):
    policy = GaussianMLPStochasticPolicy(5, 1, sigma=config["sigma"])

    value_function = NNValueFunction(obs_dim=5)

    agent = OrderedDictFlattenTransform(
        REPS(
            buffer_size=10000,
            batch_size=500,
            policy=policy,
            value_function=value_function,
            pol_opt_steps=300,
            entropy_constrained=False,
            gamma=config["gamma"],
            eta=config["eta"],
            dual_lr=config["dual_lr"],
            lr=config["lr"],
        ),
        ["observations"],
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(
        num_iterations=10,
        max_steps=config["max_steps"],
        number_rollouts=config["num_rollouts"],
    )
    policy.set_eval_mode(True)
    val_reward = trainer.validate(5, 500)

    tune.report(reward=torch.sum(torch.tensor(val_reward)).item())


analysis = tune.run(train, config=config, metric="reward", mode="max", num_samples=10)


# def eval_func(timestep):
##    action = agent.select_action(timestep)
#    return action

# viewer.launch(env, eval_func)

# t best trial: 31643_00006 with
#   reward=1969.2981863081154
# and parameters = {
#   'sigma': 1.0,
#   'num_rollouts': 10,
#   'gamma': 0.8291865624395334,
#   'eta': 0.001144007206308473,
#   'dual_lr': 0.04010895603721446,
#   'lr': 0.04668508561336196,
#   'max_steps': 200
# }
