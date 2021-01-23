from dm_env import Environment, truncation


class Trainer:
    """Trainer for running environments together with algorithms"""

    def __init__(self):
        self.env = None
        self.algo = None

    def setup(self, algo, env: Environment):
        self.env = env
        self.algo = algo

    def _obtain_episode(self, max_steps):
        timestep = self.env.reset()
        step = 0
        while not timestep.last():
            # Generate an action from the agent's policy.
            action = self.algo.select_action(timestep)

            # Step the environment.
            new_timestep = self.env.step(action)

            if step == max_steps:
                new_timestep = truncation(new_timestep.reward, new_timestep.observation)

            # Tell the agent about what just happened.
            self.algo.update(timestep, action, new_timestep)

            # Book-keeping.
            timestep = new_timestep
            step += 1

    def train(self, num_episodes, max_steps):
        """Trains the set algorithm for num_episodes and limits the steps per episode on max_steps.
        Note that the episode is returned earlier if the environment switches to done"""
        if self.algo is None or self.env is None:
            raise RuntimeError("Setup algorithm and environment before calling train.")

        for i in range(num_episodes):
            self._obtain_episode(max_steps)
