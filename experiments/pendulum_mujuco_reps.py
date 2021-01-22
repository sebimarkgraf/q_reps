import logging

import nlopt
from bsuite.baselines.experiment import run
from dm_control import suite, viewer
from dm_control.rl.control import Environment
from torch.utils.tensorboard import SummaryWriter

from qreps.fourier_features import FourierFeatures
from qreps.observation_transform import OrderedDictFlattenTransform
from qreps.policy import GaussianMLP
from qreps.reps import REPS

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

env: Environment = suite.load(domain_name="pendulum", task_name="swingup")
print(env.observation_spec())
print(env.action_spec())

writer = SummaryWriter(comment="_pendulum_reps")

feature_fn = FourierFeatures(3, 75)
policy = GaussianMLP(75, 1, minimizing_epochs=300, action_min=-1, action_max=1)

agent = OrderedDictFlattenTransform(
    REPS(
        feat_shape=(75,),
        sequence_length=1000,
        val_feature_fn=feature_fn,
        pol_feature_fn=feature_fn,
        epsilon=0.1,
        policy=policy,
        writer=writer,
        dual_optimizer_algorithm=nlopt.LD_SLSQP,
    ),
    ["orientation", "velocity"],
)

run(agent, env, num_episodes=30)

policy.set_eval_mode(True)


def eval_func(timestep):
    action = agent.select_action(timestep)
    return action


viewer.launch(env, eval_func)
