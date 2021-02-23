import logging

import torch
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.qreps import QREPS
from qreps.feature_functions.feature_concatenation import FeatureConcatenation
from qreps.feature_functions.one_hot import OneHotFeature
from qreps.policies.stochastic_table import StochasticTablePolicy
from qreps.utilities.trainer import Trainer
from qreps.valuefunctions.q_function import SimpleQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


gym_env = NChainEnv(n=5, slip=0, small=0.01, large=1.0)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

writer = SummaryWriter(comment="_chain_qreps")

feature_fn = FeatureConcatenation(
    obs_feature_fn=OneHotFeature(obs_num), act_feature_fn=OneHotFeature(act_num)
)


def pol_feature_fn(x):
    return x.long()


value_function = SimpleQFunction(
    obs_dim=obs_num, act_dim=act_num, feature_fn=feature_fn,
)

policy = StochasticTablePolicy(obs_num, act_num)

agent = QREPS(
    writer=writer,
    policy=policy,
    q_function=value_function,
    eta=5.0,
    beta=0.05,
    learner=torch.optim.SGD,
    saddle_point_steps=300,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(num_iterations=50, max_steps=30, number_rollouts=10)
policy.set_eval_mode(True)

val_reward = trainer.validate(5, 100)

logging.info(f"Val Reward {val_reward}")
writer.add_scalar("val/mean_reward", torch.mean(torch.tensor(val_reward).float()))

print("Perfect Solution should be only action 0")
for n in range(obs_num):
    act_string = ""
    for a in range(act_num):
        act_string += f"State, Action: {n, a}: {agent.q_function(torch.tensor(n), torch.tensor(a))}"
    print(
        f"State {n}, Value: {agent.value_function(torch.tensor(n))}, Action: {policy.sample(torch.tensor(n))}\n\t"
        + act_string
    )

print("Policy: ", policy._policy)
