import logging

from bsuite.baselines.experiment import run
from dm_control import suite, viewer
from dm_control.rl.control import Environment
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms import REPS
from qreps.feature_functions import FourierFeatures
from qreps.policies import GaussianMLPStochasticPolicy
from qreps.utilities.observation_transform import OrderedDictFlattenTransform
from qreps.valuefunctions import NNValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

env: Environment = suite.load(domain_name="pendulum", task_name="swingup")
print(env.observation_spec())
print(env.action_spec())

writer = SummaryWriter(comment="_pendulum_reps")

feature_fn = FourierFeatures(3, 75)

policy = GaussianMLPStochasticPolicy(75, 1, feature_fn=feature_fn)

agent = OrderedDictFlattenTransform(
    REPS(value_function=NNValueFunction(3), policy=policy, writer=writer,),
    ["orientation", "velocity"],
)

run(agent, env, num_episodes=30)

policy.set_eval_mode(True)


def eval_func(timestep):
    action = agent.select_action(timestep)
    return action


viewer.launch(env, eval_func)
