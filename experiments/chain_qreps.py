import logging

import torch
import torch.nn.functional as F
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.qreps import QREPS
from qreps.policies.stochastic_table import StochasticTablePolicy
from qreps.trainer import Trainer
from qreps.valuefunctions.q_function import NNQFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


gym_env = NChainEnv(n=5, slip=0, small=0.01, large=1.0)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

writer = SummaryWriter(comment="_chain_qreps")


def act_feature_fn(x):
    return F.one_hot(x.long(), act_num).float()


def obs_feature_fn(x):
    return F.one_hot(x.long(), obs_num).float()


def feature_fn(obs, a):
    return torch.cat((obs_feature_fn(obs), act_feature_fn(a)), dim=-1)


def pol_feature_fn(x):
    return x.long()


value_function = NNQFunction(
    obs_dim=obs_num,
    act_dim=act_num,
    feature_fn=feature_fn,
    act_feature_fn=act_feature_fn,
)


policy = StochasticTablePolicy(obs_num, act_num)

agent = QREPS(
    feature_fn=feature_fn,
    feature_dim=obs_num + act_num,
    buffer_size=5000,
    writer=writer,
    policy=policy,
    q_function=value_function,
    num_actions=act_num,
    eta=5.0,
    alpha=5.0,
    beta=0.05,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(num_iterations=5, max_steps=30, number_rollouts=10)
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
