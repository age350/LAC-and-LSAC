"""Cost-based environment used by Lyapunov Actor-Critic."""

import gymnasium as gym


class NegativeRewardToCost(gym.RewardWrapper):
    """Convert a non-positive reward into a non-negative cost."""

    def reward(self, reward):
        cost = -float(reward)
        if cost < -1e-8:
            raise ValueError(
                "The wrapped environment returned a positive reward. "
                "LAC expects a non-negative cost after conversion."
            )
        return max(0.0, cost)


def make_cost_env(env_id, seed=None):
    env = gym.make(env_id)
    env = NegativeRewardToCost(env)
    env = gym.wrappers.FlattenObservation(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

