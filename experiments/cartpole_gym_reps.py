import logging

import gym
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.reps import REPS
from qreps.policies.policy import CategoricalMLP
from qreps.trainer import Trainer
from qreps.valuefunctions.value_functions import SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

gym_env = gym.make("CartPole-v0")
env = gym_wrapper.DMEnvFromGym(gym_env)
print(env.observation_spec())
print(env.action_spec())

num_obs = env.observation_spec().shape[0]
writer = SummaryWriter(comment="_cartpole_gym_reps")
policy = CategoricalMLP(num_obs, 2)

agent = REPS(
    buffer_size=5000,
    batch_size=50,
    epsilon=1.0,
    policy=policy,
    value_function=SimpleValueFunction(num_obs),
    gamma=1.0,
    writer=writer,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(10, 200, number_rollouts=15)

policy.set_eval_mode(True)

val_rewards = trainer.validate(5, 2000, render=True)
logging.info(f"Validation rewards: {val_rewards}")
