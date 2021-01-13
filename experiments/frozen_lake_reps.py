import logging

import gym
import torch
from bsuite.baselines.experiment import run
from bsuite.utils import gym_wrapper

from qreps.policy import DiscreteStochasticPolicy
from qreps.reps import REPS

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

gym_env = gym.make("FrozenLake-v0")
env = gym_wrapper.DMEnvFromGym(gym_env)

obs_num = env.observation_spec().num_values


def feature_fn(x):
    return torch.nn.functional.one_hot(torch.tensor(x, dtype=torch.int64), obs_num).to(
        torch.float32
    )


def pol_feature_fn(x):
    return torch.tensor(x, dtype=torch.int64)


agent = REPS(
    feat_shape=(obs_num,),
    sequence_length=20,
    val_feature_fn=feature_fn,
    pol_feature_fn=pol_feature_fn,
    epsilon=1e-12,
    policy=DiscreteStochasticPolicy(obs_num, env.action_spec().num_values),
)


for i in range(20):
    print("Iteration", i)
    run(agent, env, num_episodes=50)

    timestep = env.reset()
    while not timestep.last():
        # Generate an action from the agent's policy.
        action = agent.select_action(timestep)

        # Step the environment.
        new_timestep = env.step(action)

        # Book-keeping.
        timestep = new_timestep
        gym_env.render()
