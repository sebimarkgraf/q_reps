import torch
import torch.nn.functional as F
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from ray import tune

from qreps.algorithms.reps import REPS
from qreps.policies.stochastic_table import StochasticTablePolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions.value_functions import SimpleValueFunction

gym_env = NChainEnv(n=5, slip=0.2, small=0.1)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

config = {
    "num_rollouts": tune.grid_search([3, 5, 10, 15]),
    "gamma": tune.uniform(0.8, 1.0),
    "eta": tune.loguniform(2e-4, 2e-1, 10),
    "dual_lr": tune.loguniform(1e-3, 1e-1),
    "lr": tune.loguniform(1e-3, 1e-1),
}


def train(config: dict):
    def feature_fn(x):
        return F.one_hot(x.long(), obs_num).float()

    value_function = SimpleValueFunction(obs_dim=obs_num, feature_fn=feature_fn)

    policy = StochasticTablePolicy(obs_num, act_num)

    agent = REPS(
        buffer_size=5000,
        policy=policy,
        value_function=value_function,
        gamma=config["gamma"],
        eta=config["eta"],
        dual_lr=config["dual_lr"],
        lr=config["lr"],
    )

    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(
        num_iterations=10, max_steps=30, number_rollouts=config["num_rollouts"]
    )
    policy.set_eval_mode(True)
    val_reward = trainer.validate(5, 100)
    tune.report(reward=torch.sum(torch.tensor(val_reward)).item())


analysis = tune.run(train, config=config, metric="reward", mode="max", num_samples=30)


print(analysis.best_config)
