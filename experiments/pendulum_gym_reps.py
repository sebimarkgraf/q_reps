import logging

import gym
import nlopt
from bsuite.baselines.experiment import run
from bsuite.utils import gym_wrapper
from dm_control import suite, viewer
from dm_control.rl.control import Environment
from torch.utils.tensorboard import SummaryWriter

from qreps.fourier_features import FourierFeatures
from qreps.observation_transform import OrderedDictFlattenTransform
from qreps.policy import CategoricalMLP, GaussianMLP
from qreps.reps import REPS

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
    minimizing_epochs=300,
    action_min=env.action_spec().minimum[0],
    action_max=env.action_spec().maximum[0],
)

agent = REPS(
    feat_shape=(75,),
    sequence_length=1000,
    val_feature_fn=feature_fn,
    pol_feature_fn=feature_fn,
    epsilon=0.1,
    policy=policy,
    writer=writer,
    dual_optimizer_algorithm=nlopt.LD_SLSQP,
)


run(agent, env, num_episodes=30)

policy.set_eval_mode(True)

val_rewards = []
for i in range(5):
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
