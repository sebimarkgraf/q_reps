import gym
from bsuite.baselines.experiment import run
from bsuite.utils import gym_wrapper

from qreps.policy import DiscreteStochasticPolicy
from qreps.reps import REPS

env = gym_wrapper.DMEnvFromGym(gym.make("FrozenLake-v0"))
agent = REPS(
    obs_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    sequence_length=50,
    epsilon=1e-5,
    policy=DiscreteStochasticPolicy(
        env.observation_spec().num_values, env.action_spec().num_values
    ),
)


run(agent, env, num_episodes=500)
