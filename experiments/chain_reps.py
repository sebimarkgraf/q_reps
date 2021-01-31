import logging

import torch
import torch.nn.functional as F
from bsuite.utils import gym_wrapper
from gym.envs.toy_text import NChainEnv
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.reps import REPS
from qreps.policies.stochastic_table import StochasticTablePolicy
from qreps.policies.value_function_policy import ValueFunctionPolicy
from qreps.trainer import Trainer
from qreps.valuefunctions.value_functions import SimpleValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


gym_env = NChainEnv(n=5, slip=0, small=0.1)
env = gym_wrapper.DMEnvFromGym(gym_env)
obs_num = env.observation_spec().num_values
act_num = env.action_spec().num_values

writer = SummaryWriter(comment="_chain_reps")


def feature_fn(x):
    return F.one_hot(x.long(), obs_num).float()


def pol_feature_fn(x):
    return x.long()


value_function = SimpleValueFunction(obs_num, feature_fn)

transition_model = torch.tensor([[1, 0], [2, 0], [3, 0], [4, 0], [4, 0]])

# policy = ValueFunctionPolicy(
#    n_actions=act_num,
#    value_function=value_function,
#    transition_model=transition_model,
#    eps=0.5
# )

policy = StochasticTablePolicy(obs_num, act_num)

agent = REPS(
    buffer_size=5000,
    batch_size=50,
    writer=writer,
    epsilon=2.0,
    policy=policy,
    center_advantages=False,
    value_function=value_function,
    gamma=0.9,
    optimize_policy=True,
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(num_iterations=10, max_steps=100, number_rollouts=5)
policy.set_eval_mode(True)

val_reward = trainer.validate(5, 100)

logging.info(f"Val Reward {val_reward}")
writer.add_scalar("val/mean_reward", torch.mean(torch.tensor(val_reward).float()))

print("Perfect Solution should be only action 0")
for n in range(obs_num):
    print(
        f"State {n}, Value: {agent.value_function(torch.tensor(n))}, Action: {policy.sample(torch.tensor(n))}"
    )

print("Policy: ", policy._policy)
