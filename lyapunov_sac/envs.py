"""Environment wrapper that exposes reward and a positive control cost."""

import gymnasium as gym


class RewardWithCost(gym.Wrapper):
    """Keep the original reward and add a non-negative cost to info."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cost = -float(reward)
        if cost < -1e-8:
            raise ValueError(
                "The default cost conversion requires non-positive rewards. "
                "For another environment, define its positive control cost here."
            )
        info = dict(info)
        info["cost"] = max(0.0, cost)
        return obs, float(reward), terminated, truncated, info


def make_env(env_id, seed=None):
    """Create the continuous-control environment used by LSAC."""
    env = gym.make(env_id)
    env = RewardWithCost(env)
    env = gym.wrappers.FlattenObservation(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

