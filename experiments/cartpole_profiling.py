import sys
import time

import gym
import torch

from qreps.policies.qreps_policy import QREPSPolicy

sys.path.append("../")
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import QREPS, REPS
from qreps.algorithms.sampler import BestResponseSampler
from qreps.feature_functions import FeatureConcatenation, NNFeatures, OneHotFeature
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.utilities.util import set_seed
from qreps.valuefunctions import SimpleQFunction, SimpleValueFunction

reps_config = {
    "discount": 0.99,
    "eta": 0.4515,
    "dual_lr": 2e-2,
    "policy_lr": 2e-5,
    "entropy_constrained": False,
    "dual_opt_steps": 300,
    "policy_opt_steps": 300,
}

qreps_config = {
    "eta": 0.5,
    "beta": 2e-2,
    "saddle_point_steps": 300,
    "policy_opt_steps": 300,
    "policy_lr": 2e-5,
    "discount": 0.99,
}


def create_env(seed):
    gym_env = gym.make("CartPole-v0")
    gym_env.seed(seed)
    env = gym_wrapper.DMEnvFromGym(gym_env)
    num_obs = env.observation_spec().shape[0]
    num_act = env.action_spec().num_values
    return env, num_obs, num_act


def create_agent(algo, writer, config, num_obs, num_act):
    policy = CategoricalMLP(obs_shape=num_obs, act_shape=num_act)
    FEAT_DIM = 200
    feature_fn = NNFeatures(num_obs, feat_dim=FEAT_DIM)

    if algo == "reps":
        value_function = SimpleValueFunction(obs_dim=FEAT_DIM, feature_fn=feature_fn)
        return REPS(
            policy=policy,
            value_function=value_function,
            writer=writer,
            reward_transformer=lambda r: r / 1000,
            **config,
        )
    elif algo == "qreps":
        q_function = SimpleQFunction(
            obs_dim=FEAT_DIM, act_dim=num_act, feature_fn=feature_fn,
        )

        return QREPS(
            writer=writer,
            policy=policy,
            q_function=q_function,
            learner=torch.optim.Adam,
            sampler=BestResponseSampler,
            reward_transformer=lambda r: r / 1000,
            **config,
        )
    elif algo == "qreps_nonparametric":
        q_function = SimpleQFunction(
            obs_dim=FEAT_DIM, act_dim=num_act, feature_fn=feature_fn,
        )
        policy = QREPSPolicy(q_function=q_function, temp=config["eta"])

        return QREPS(
            writer=writer,
            policy=policy,
            q_function=q_function,
            learner=torch.optim.Adam,
            sampler=BestResponseSampler,
            reward_transformer=lambda r: r / 1000,
            optimize_policy=False,
            **config,
        )


def evaluate(agent, env):
    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=30, max_steps=200, number_rollouts=5)


def main():
    timestamp = time.time()
    for algo in ["reps", "qreps", "qreps_nonparametric"]:
        for it in range(5):
            set_seed(it)
            print(f"Runing {algo}")
            config = reps_config if algo == "reps" else qreps_config
            wandb.init(
                project="qreps",
                entity="sebimarkgraf",
                sync_tensorboard=True,
                tags=["cartpole_profiling", algo, f"profile_run_{timestamp}"],
                group=f"cartpole_profiling_{algo}",
                config=config,
            )
            writer = SummaryWriter()
            env, num_obs, num_act = create_env(seed=it)
            agent = create_agent(algo, writer, config, num_obs, num_act)
            evaluate(agent, env)
            writer.close()
            wandb.finish()


if __name__ == "__main__":
    main()
