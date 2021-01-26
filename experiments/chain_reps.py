import logging

import torch
import torch.nn.functional as F
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from torch.utils.tensorboard import SummaryWriter

from qreps.policy import DiscreteStochasticPolicy
from qreps.reps import REPS
from qreps.trainer import Trainer

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


gym_env = NChainEnv(n=2, slip=0, small=0.1)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values

writer = SummaryWriter(comment="_chain_reps")


def feature_fn(x):
    return F.one_hot(torch.tensor(x).long(), obs_num).float()


def pol_feature_fn(x):
    return torch.tensor(x).long()


policy = DiscreteStochasticPolicy(obs_num, env.action_spec().num_values)

agent = REPS(
    feat_shape=(obs_num,),
    buffer_size=1000,
    batch_size=300,
    writer=writer,
    val_feature_fn=feature_fn,
    pol_feature_fn=pol_feature_fn,
    epsilon=1e-5,
    policy=policy,
    center_advantages=False,
)

trainer = Trainer()
trainer.setup(agent, env)

trainer.train(30, 300)

policy.set_eval_mode(True)

val_reward = trainer.validate(5, 100)

logging.info(f"Val Reward {val_reward}")
writer.add_scalar("val/mean_reward", torch.mean(torch.tensor(val_reward).float()))

print("Perfect Solution should be only action 0")
for n in range(obs_num):
    print(
        f"State {n}, Value: {agent.theta.dot(feature_fn(n))}, Action: {policy.sample(pol_feature_fn(n))}"
    )

print("Policy: ", policy._policy)
