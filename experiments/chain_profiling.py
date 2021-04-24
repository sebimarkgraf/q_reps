import sys
import time

from qreps.policies.qreps_policy import QREPSPolicy

sys.path.append("../")

import torch
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import QREPS, REPS
from qreps.feature_functions import FeatureConcatenation, OneHotFeature
from qreps.policies import StochasticTablePolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import SimpleQFunction, SimpleValueFunction

gym_env = NChainEnv()
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

NUM_ITERATIONS = 5

reps_config = {
    "discount": 0.99,
    "eta": 2.0,
    "dual_lr": 0.07,
    "policy_lr": 0.005,
    "entropy_constrained": True,
    "dual_opt_steps": 300,
    "policy_opt_steps": 300,
}


qreps_config = {
    "eta": 5.0,
    "beta": 0.05,
    "saddle_point_steps": 300,
    "policy_opt_steps": 300,
    "policy_lr": 0.04,
    "discount": 1.0,
    "grad_samples": 1,
}


def create_agent(algo, writer, config):

    policy = StochasticTablePolicy(obs_num, act_num)
    if algo == "reps":
        value_function = SimpleValueFunction(
            obs_dim=obs_num, feature_fn=OneHotFeature(obs_num)
        )
        return REPS(
            writer=writer, policy=policy, value_function=value_function, **config
        )
    elif algo == "qreps":

        feature_fn = OneHotFeature(obs_num)
        value_function = SimpleQFunction(
            obs_dim=obs_num, act_dim=act_num, feature_fn=feature_fn,
        )
        return QREPS(
            writer=writer,
            policy=policy,
            q_function=value_function,
            learner=torch.optim.SGD,
            **config,
        )
    elif algo == "qreps_nonparametric":

        feature_fn = OneHotFeature(obs_num)
        value_function = SimpleQFunction(
            obs_dim=obs_num, act_dim=act_num, feature_fn=feature_fn,
        )
        policy = QREPSPolicy(q_function=value_function)
        return QREPS(
            writer=writer,
            policy=policy,
            q_function=value_function,
            learner=torch.optim.SGD,
            **config,
        )


def evaluate(agent):
    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=10, max_steps=200, number_rollouts=1)


def main():
    timestamp = time.time()
    for algo in ["reps", "qreps", "qreps_nonparametric"]:
        for it in range(NUM_ITERATIONS):
            print(f"Runing {algo}")
            config = reps_config if algo == "reps" else qreps_config
            wandb.init(
                project="qreps",
                entity="sebimarkgraf",
                sync_tensorboard=True,
                tags=["chain_profiling", algo, f"profile_run_{timestamp}"],
                group=f"chain_profiling_{algo}",
                config=config,
            )
            writer = SummaryWriter()
            agent = create_agent(algo, writer, config)
            evaluate(agent)
            writer.close()
            wandb.finish()


if __name__ == "__main__":
    main()
