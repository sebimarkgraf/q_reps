import logging
import time

import gym
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.reps import REPS
from qreps.policies.categorical_mlp import CategoricalMLP
from qreps.policies.gaussian_mlp import GaussianMLPStochasticPolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions.value_functions import NNValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


timestamp = time.time()
gym_env = gym.make("Pendulum-v0")
# gym_env =  gym.wrappers.Monitor(gym_env, directory=f"./frozen_lake_{timestamp}")
env = gym_wrapper.DMEnvFromGym(gym_env)
print(env.observation_spec())
print(env.action_spec())

num_obs = env.observation_spec().shape[0]
writer = SummaryWriter(comment="_pendulum_gym_reps")
policy = GaussianMLPStochasticPolicy(num_obs, 1)

agent = REPS(
    batch_size=50,
    policy=policy,
    value_function=NNValueFunction(obs_dim=num_obs),
    writer=writer,
    eta=1.0,
    entropy_constrained=True,
    lr=5e-4,
    dual_lr=5e-4,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(10, 200, number_rollouts=15)

policy.set_eval_mode(True)

val_rewards = trainer.validate(5, 200)

logging.info(f"Validation rewards: {val_rewards}")
