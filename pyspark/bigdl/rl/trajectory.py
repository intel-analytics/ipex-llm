import numpy as np

class Trajectory(object):

    fields = ["observations", "actions", "rewards", "terminal"]

    def __init__(self):
        self.data = {k: [] for k in self.fields}
        self.last_r = 0.0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k] += [v]

    def is_terminal(self):
        return self.data["terminal"][-1]


class Sampler(object):

    def __init__(self, env, horizon=None):
        self.horizon = horizon
        self.env = env
        self.last_obs = env.reset()

    def get_data(self, policy, max_steps):
        return self._run_policy(
            self.env, policy, max_steps, self.horizon)

    def _run_policy(self, env, policy, max_steps, horizon):
        length = 0

        traj = Trajectory()

        for _ in range(max_steps):
            action_distribution = policy.forward(self.last_obs)
            action = np.random.multinomial(1, action_distribution).argmax()
            observation, reward, terminal, info = env.step(action)

            length += 1
            if length >= horizon:
                terminal = True

            # Collect the experience.
            traj.add(observations=self.last_obs,
                     actions=action,
                     rewards=reward,
                     terminal=terminal)

            self.last_obs = observation

            if terminal:
                self.last_obs = env.reset()
                break
        return traj
