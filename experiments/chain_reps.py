import logging

import torch
import torch.nn.functional as F
from bsuite.baselines.experiment import run
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from envs.nchain import NChainEnv
from qreps.policy import DiscreteStochasticPolicy
from qreps.reps import REPS

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


gym_env = NChainEnv(n=10, slip=0, small=0.1, max_steps=200)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values

writer = SummaryWriter(comment="chain_reps")


def feature_fn(x):
    return F.one_hot(torch.tensor(x, dtype=torch.int64), obs_num).to(torch.float32)


def pol_feature_fn(x):
    return torch.tensor(x, dtype=torch.int64)


policy = DiscreteStochasticPolicy(obs_num, env.action_spec().num_values)

agent = REPS(
    feat_shape=(obs_num,),
    sequence_length=1000,
    writer=writer,
    val_feature_fn=feature_fn,
    pol_feature_fn=pol_feature_fn,
    epsilon=1e-5,
    policy=policy,
    center_advantages=False,
)

run(agent, env, num_episodes=100)


policy.set_eval_mode(True)

val_reward = 0
for i in range(5):
    timestep = env.reset()
    while not timestep.last():
        # Generate an action from the agent's policy.
        action = agent.select_action(timestep)
        # Step the environment.
        new_timestep = env.step(action)

        # Book-keeping.
        timestep = new_timestep
        val_reward += timestep.reward

logging.info(f"Val Reward {val_reward}")
writer.add_scalar("val/reward", val_reward)

for n in range(obs_num):
    print(f"State {n}, Value: {agent.theta.dot(feature_fn(n))}")

print("Policy", policy._policy)
