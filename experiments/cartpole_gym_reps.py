import logging

import gym
import nlopt
from bsuite.baselines.experiment import run
from bsuite.utils import gym_wrapper
from torch.utils.tensorboard import SummaryWriter

from qreps.fourier_features import FourierFeatures
from qreps.policy import CategoricalMLP
from qreps.reps import REPS
from qreps.util import to_torch

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

gym_env = gym.make("CartPole-v0")
env = gym_wrapper.DMEnvFromGym(gym_env)
print(env.observation_spec())
print(env.action_spec())

writer = SummaryWriter(comment="_cartpole_gym_reps")

feature_fn = to_torch
policy = CategoricalMLP(env.observation_spec().shape[0], 2, minimizing_epochs=300,)

agent = REPS(
    feat_shape=(4,),
    sequence_length=1000,
    val_feature_fn=feature_fn,
    pol_feature_fn=feature_fn,
    epsilon=0.3,
    policy=policy,
    writer=writer,
    dual_optimizer_algorithm=nlopt.LD_SLSQP,
)


run(agent, env, num_episodes=1000)

policy.set_eval_mode(True)

val_rewards = []
for i in range(10):
    timestep = env.reset()
    val_reward = 0
    while not timestep.last():
        # Generate an action from the agent's policy.
        action = agent.select_action(timestep)
        # Step the environment.
        new_timestep = env.step(action)

        # Book-keeping.
        timestep = new_timestep
        val_reward += timestep.reward
        gym_env.render()
    val_rewards.append(val_reward)

print(f"Val reward: {val_rewards}")
