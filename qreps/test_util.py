import bsuite
import gym
from bsuite.utils import gym_wrapper

from qreps.util import num_from_spec


def test_gym_envs_num_values():
    env = gym_wrapper.DMEnvFromGym(gym.make("FrozenLake-v0"))
    num_obs = num_from_spec(env.observation_spec())
    num_action = num_from_spec(env.action_spec())

    assert num_obs == 16
    assert num_action == 4


def test_bsuite_envs_num_values():
    env = bsuite.load_from_id("cartpole/1")
    num_obs = num_from_spec(env.observation_spec())
    num_actions = num_from_spec(env.action_spec())

    assert num_obs == 10
    assert num_actions == 4
