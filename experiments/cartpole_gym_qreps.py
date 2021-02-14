import logging
import time

import gym
import torch
import torch.nn.functional as F
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.qreps import QREPS
from qreps.algorithms.reps import REPS
from qreps.policies.categorical_mlp import CategoricalMLP
from qreps.trainer import Trainer
from qreps.valuefunctions.value_functions import SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


timestamp = time.time()
gym_env = gym.make("CartPole-v0")
# gym_env =  gym.wrappers.Monitor(gym_env, directory=f"./frozen_lake_{timestamp}", video_callable=lambda x: x % 100)
env = gym_wrapper.DMEnvFromGym(gym_env)
print(env.observation_spec())
print(env.action_spec())

num_obs = env.observation_spec().shape[0]
num_act = env.action_spec().num_values
writer = SummaryWriter(comment="_cartpole_gym_qreps")
policy = CategoricalMLP(num_obs, 2)


def act_feature_fn(x):
    return F.one_hot(x.long(), num_act).float()


def obs_feature_fn(x):
    return x.float()


def feature_fn(obs, a):
    return torch.cat((obs_feature_fn(obs), act_feature_fn(a)), dim=-1)


agent = QREPS(
    feature_fn=feature_fn,
    feature_dim=num_obs + num_act,
    buffer_size=5000,
    writer=writer,
    policy=policy,
    eta=0.1,
    alpha=0.1,
    beta=0.05,
    num_actions=num_act,
    q_function=None,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(100, 200, number_rollouts=15)

policy.set_eval_mode(True)

val_rewards = trainer.validate(5, 200)

timestep = env.reset()
step = 0

while not timestep.last():
    # Generate an action from the agent's policy.
    action = agent.select_action(timestep)
    # Step the environment.
    new_timestep = env.step(action)
    # Book-keeping.
    timestep = new_timestep


logging.info(f"Validation rewards: {val_rewards}")
