import logging
import time

import gym
import torch
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import QREPS
from qreps.algorithms.sampler import BestResponseSampler
from qreps.feature_functions import FeatureConcatenation, NNFeatures, OneHotFeature
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import SimpleQFunction

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


feature_fn = FeatureConcatenation(
    obs_feature_fn=NNFeatures(num_obs, feat_dim=200),
    act_feature_fn=OneHotFeature(num_classes=num_act),
)

q_function = SimpleQFunction(obs_dim=200, act_dim=num_act, feature_fn=feature_fn)

agent = QREPS(
    writer=writer,
    policy=policy,
    eta=0.01,
    beta=0.08,
    discount=0.99,
    q_function=q_function,
    learner=torch.optim.Adam,
    sampler=BestResponseSampler,
    policy_lr=5e-4,
    policy_opt_steps=500,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(30, 200, number_rollouts=5)

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
