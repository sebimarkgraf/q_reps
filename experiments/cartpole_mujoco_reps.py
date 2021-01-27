import logging

from dm_control import viewer
from dm_control.rl.control import Environment
from dm_control.suite.cartpole import balance
from torch.utils.tensorboard import SummaryWriter

from qreps.observation_transform import OrderedDictFlattenTransform
from qreps.policy import GaussianMLP
from qreps.reps import REPS
from qreps.trainer import Trainer
from qreps.util import to_torch
from qreps.value_functions import SimpleValueFunction

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

feature_fn = to_torch
pol_feature_fn = to_torch
policy = GaussianMLP(
    5,
    1,
    minimizing_epochs=300,
    sigma=1.0,
    action_min=env.action_spec().minimum[0],
    action_max=env.action_spec().maximum[0],
)

agent = OrderedDictFlattenTransform(
    REPS(
        buffer_size=3000,
        batch_size=500,
        val_feature_fn=feature_fn,
        pol_feature_fn=pol_feature_fn,
        epsilon=0.1,
        policy=policy,
        writer=writer,
        value_function=SimpleValueFunction(5),
    ),
    ["observations"],
)

trainer = Trainer()
trainer.setup(agent, env)
trainer.train(200, 2000)

policy.set_eval_mode(True)

val_rewards = trainer.validate(5, 2000)
logging.info(f"Validation rewards: {val_rewards}")


def eval_func(timestep):
    action = agent.select_action(timestep)
    return action


viewer.launch(env, eval_func)
