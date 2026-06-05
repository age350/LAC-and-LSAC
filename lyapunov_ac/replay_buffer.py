"""FIFO replay buffer for off-policy LAC training."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.obs = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.obs_next = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.action = np.zeros(
            (capacity, act_dim), dtype=np.float32
        )
        self.cost = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.device = device
        self.pointer = 0
        self.size = 0

    def store(self, obs, action, cost, obs_next, done):
        self.obs[self.pointer] = obs
        self.action[self.pointer] = action
        self.cost[self.pointer] = cost
        self.obs_next[self.pointer] = obs_next
        self.done[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(
            0, self.size, size=batch_size
        )
        batch = {
            "obs": self.obs[indices],
            "action": self.action[indices],
            "cost": self.cost[indices],
            "obs_next": self.obs_next[indices],
            "done": self.done[indices],
        }
        return {
            key: torch.as_tensor(
                value,
                dtype=torch.float32,
                device=self.device,
            )
            for key, value in batch.items()
        }

