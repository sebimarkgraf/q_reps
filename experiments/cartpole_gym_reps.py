import logging
import time

import gym
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import REPS
from qreps.feature_functions import NNFeatures
from qreps.policies import CategoricalMLP
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions import NNValueFunction, SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


timestamp = time.time()
gym_env = gym.make("CartPole-v0")
# gym_env =  gym.wrappers.Monitor(gym_env, directory=f"./frozen_lake_{timestamp}")
env = gym_wrapper.DMEnvFromGym(gym_env)
print(env.observation_spec())
print(env.action_spec())

num_obs = env.observation_spec().shape[0]
writer = SummaryWriter(comment="_cartpole_gym_reps")
policy = CategoricalMLP(num_obs, 2)


feature_fn = NNFeatures(num_obs, feat_dim=200)

agent = REPS(
    batch_size=50,
    policy=policy,
    value_function=SimpleValueFunction(obs_dim=200, feature_fn=feature_fn),
    writer=writer,
    eta=5.0,
    entropy_constrained=False,
    policy_lr=5e-4,
    dual_lr=0.01,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(30, 200, number_rollouts=5)

policy.set_eval_mode(True)

val_rewards = trainer.validate(5, 200)

logging.info(f"Validation rewards: {val_rewards}")
