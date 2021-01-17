import logging

from dm_control import suite, viewer
from dm_control.rl.control import Environment
from dm_control.suite.cartpole import balance
from torch.utils.tensorboard import SummaryWriter

from qreps.fourier_features import FourierFeatures
from qreps.observation_transform import OrderedDictFlattenTransform
from qreps.policy import GaussianMLP
from qreps.reps import REPS

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

env: Environment = balance(
    time_limit=3.0, environment_kwargs={"flat_observation": True}
)
# env: Environment = suite.load(domain_name="cartpole", task_name="balance", visualize_reward=True, environment_kwargs={'flat_observation': True, 'time_limit': 5.0})
print(env.observation_spec())
print(env.action_spec())
print(env.discount_spec())

writer = SummaryWriter(comment="_mujuco_reps")

feature_fn = FourierFeatures(5, 75)
policy = GaussianMLP(
    75,
    1,
    minimizing_epochs=300,
    action_min=env.action_spec().minimum[0],
    action_max=env.action_spec().maximum[0],
)

agent = OrderedDictFlattenTransform(
    REPS(
        feat_shape=(75,),
        sequence_length=2000,
        val_feature_fn=feature_fn,
        pol_feature_fn=feature_fn,
        epsilon=1e-5,
        policy=policy,
        writer=writer,
    ),
    ["observations"],
)


# Train loop
for _ in range(10):
    # Run an episode.
    timestep = env.reset()
    while not timestep.last():
        # Generate an action from the agent's policy.
        action = agent.select_action(timestep)

        # Step the environment.
        new_timestep = env.step(action)

        # Tell the agent about what just happened.
        agent.update(timestep, action, new_timestep)

        # Book-keeping.
        timestep = new_timestep


# policy.set_eval_mode(True)


def eval_func(timestep):
    action = agent.select_action(timestep)
    return action


viewer.launch(env, eval_func)
