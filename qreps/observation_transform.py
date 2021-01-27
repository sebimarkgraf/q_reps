import dm_env
import numpy as np
from bsuite.baselines import base
from bsuite.baselines.base import Action


class OrderedDictFlattenTransform(base.Agent):
    def __init__(self, agent: base.Agent, identifiers: list):
        self._agent = agent
        self.identifiers = identifiers

    def obs_transform(self, observation):
        arrays = [observation[identifier] for identifier in self.identifiers]
        return np.concatenate(arrays)

    def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
        timestep_1 = dm_env.TimeStep(
            timestep.step_type,
            timestep.reward,
            timestep.discount,
            self.obs_transform(timestep.observation),
        )
        return self._agent.select_action(timestep_1)

    def update(
        self, timestep: dm_env.TimeStep, action: Action, new_timestep: dm_env.TimeStep,
    ) -> None:
        timestep_1 = dm_env.TimeStep(
            timestep.step_type,
            timestep.reward,
            timestep.discount,
            self.obs_transform(timestep.observation),
        )
        new_timestep_1 = dm_env.TimeStep(
            new_timestep.step_type,
            new_timestep.reward,
            new_timestep.discount,
            self.obs_transform(new_timestep.observation),
        )
        self._agent.update(timestep_1, action, new_timestep_1)

    def update_policy(self, iteration):
        self._agent.update_policy(iteration)
