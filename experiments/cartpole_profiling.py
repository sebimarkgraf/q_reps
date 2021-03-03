import sys
import time

import gym
import torch

from qreps.algorithms.sampler import BestResponseSampler

sys.path.append("../")
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

import wandb
from qreps.algorithms import QREPS, REPS
from qreps.feature_functions import FeatureConcatenation, NNFeatures, OneHotFeature
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import SimpleQFunction, SimpleValueFunction

torch.manual_seed(1234)

gym_env = gym.make("CartPole-v0")
env = gym_wrapper.DMEnvFromGym(gym_env)
num_obs = env.observation_spec().shape[0]
num_act = env.action_spec().num_values

reps_config = {
    "discount": 1.0,
    "eta": 5.0,
    "dual_lr": 0.01,
    "policy_lr": 5e-4,
    "entropy_constrained": False,
    "dual_opt_steps": 300,
}


qreps_config = {
    "eta": 0.01,
    "beta": 0.08,
    "saddle_point_steps": 300,
    "policy_opt_steps": 600,
    "policy_lr": 2e-2,
    "discount": 0.99,
}


def create_agent(algo, writer, config):

    policy = CategoricalMLP(obs_shape=num_obs, act_shape=num_act)
    FEAT_DIM = 200
    obs_feature_fn = NNFeatures(num_obs, feat_dim=FEAT_DIM)

    if algo == "reps":
        value_function = SimpleValueFunction(
            obs_dim=FEAT_DIM, feature_fn=obs_feature_fn
        )
        return REPS(
            policy=policy, value_function=value_function, writer=writer, **config
        )
    elif algo == "qreps":
        feature_fn = FeatureConcatenation(
            obs_feature_fn=obs_feature_fn,
            act_feature_fn=OneHotFeature(num_classes=num_act),
        )
        q_function = SimpleQFunction(
            obs_dim=FEAT_DIM, act_dim=num_act, feature_fn=feature_fn,
        )

        return QREPS(
            writer=writer,
            policy=policy,
            q_function=q_function,
            learner=torch.optim.Adam,
            sampler=BestResponseSampler,
            **config,
        )


def evaluate(agent):
    trainer = Trainer()
    trainer.setup(agent, env)
    trainer.train(num_iterations=30, max_steps=200, number_rollouts=5)


def main():
    timestamp = time.time()
    for algo in ["reps", "qreps"]:
        for it in range(10):
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
            agent = create_agent(algo, writer, config)
            evaluate(agent)
            writer.close()
            wandb.finish()


if __name__ == "__main__":
    main()
