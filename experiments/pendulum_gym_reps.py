import logging

import gym
from bsuite.utils import gym_wrapper
from dm_control.rl.control import Environment
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.reps import REPS
from qreps.fourier_features import FourierFeatures
from qreps.policies.policy import GaussianMLP
from qreps.trainer import Trainer
from qreps.valuefunctions.value_functions import SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

gym_env: Environment = gym.make("Pendulum-v0")
env = gym_wrapper.DMEnvFromGym(gym_env)
print(env.observation_spec())
print(env.action_spec())

writer = SummaryWriter(comment="_pendulum_gym_reps")

NUM_FEATURES = 75
feature_fn = FourierFeatures(env.observation_spec().shape[0], NUM_FEATURES)
policy = GaussianMLP(
    NUM_FEATURES,
    env.action_spec().shape[0],
    action_min=env.action_spec().minimum[0],
    action_max=env.action_spec().maximum[0],
    feature_fn=feature_fn,
)

agent = REPS(
    buffer_size=2000,
    batch_size=100,
    epsilon=1.0,
    policy=policy,
    writer=writer,
    value_function=SimpleValueFunction(NUM_FEATURES, feature_fn=feature_fn),
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(100, 200, number_rollouts=10)

policy.set_eval_mode(True)

val_rewards = trainer.validate(5, 2000)
logging.info(f"Validation rewards: {val_rewards}")
