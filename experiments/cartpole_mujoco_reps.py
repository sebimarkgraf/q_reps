import logging

from dm_control import viewer
from dm_control.rl.control import Environment
from dm_control.suite.cartpole import balance
from torch.utils.tensorboard import SummaryWriter

from qreps.algorithms.reps import REPS
from qreps.observation_transform import OrderedDictFlattenTransform
from qreps.policies.gaussian_mlp import GaussianMLPStochasticPolicy
from qreps.trainer import Trainer
from qreps.valuefunctions.value_functions import NNValueFunction

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

env: Environment = balance(
    time_limit=10.0, environment_kwargs={"flat_observation": True}
)
print(env.observation_spec())
print(env.action_spec())
print(env.discount_spec())
writer = SummaryWriter(comment="_mujuco_reps")

policy = GaussianMLPStochasticPolicy(
    5,
    1,
    sigma=1.0,
    action_min=env.action_spec().minimum[0],
    action_max=env.action_spec().maximum[0],
)

agent = OrderedDictFlattenTransform(
    REPS(
        buffer_size=10000,
        batch_size=500,
        policy=policy,
        writer=writer,
        value_function=NNValueFunction(5),
        pol_opt_steps=300,
        entropy_constrained=False,
    ),
    ["observations"],
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(30, 2000, number_rollouts=3)

policy.set_eval_mode(True)

val_rewards = trainer.validate(5, 2000)
logging.info(f"Validation rewards: {val_rewards}")


def eval_func(timestep):
    action = agent.select_action(timestep)
    return action


viewer.launch(env, eval_func)
