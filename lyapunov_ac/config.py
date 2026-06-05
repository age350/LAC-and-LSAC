"""Training configuration for episode-based LAC."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfig:
    env_id: str = "Pendulum-v1"
    episodes: int = 100
    start_steps: int = 0
    update_after: int = 1000
    update_every: int = 100
    gradient_steps: int = 100
    batch_size: int = 256
    replay_size: int = 1_000_000
    hidden_sizes: Tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.99
    alpha3: float = 0.2
    labda: float = 0.99
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 1e-4
    labda_lr: float = 3e-4
    actor_lr_final: float = 1e-10
    critic_lr_final: float = 1e-10
    alpha_lr_final: float = 1e-10
    labda_lr_final: float = 1e-10
    lr_decay_type: str = "linear"
    test_episodes: int = 1
    eval_interval: int = 10
    reward_smoothing: int = 10
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "runs"

    def validate(self):
        if self.episodes <= 0:
            raise ValueError("episodes must be positive.")
        if self.start_steps < 0:
            raise ValueError("start_steps cannot be negative.")
        if self.update_after < 1:
            raise ValueError("update_after must be at least 1.")
        if self.update_every <= 0 or self.gradient_steps <= 0:
            raise ValueError("update_every and gradient_steps must be positive.")
        if self.batch_size <= 0 or self.replay_size < self.batch_size:
            raise ValueError("replay_size must be at least batch_size.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.polyak <= 1.0:
            raise ValueError("polyak must be in [0, 1].")
        if self.alpha <= 0.0 or self.labda <= 0.0:
            raise ValueError("alpha and labda must be positive.")
        learning_rates = (
            self.actor_lr,
            self.critic_lr,
            self.alpha_lr,
            self.labda_lr,
            self.actor_lr_final,
            self.critic_lr_final,
            self.alpha_lr_final,
            self.labda_lr_final,
        )
        if any(learning_rate <= 0.0 for learning_rate in learning_rates):
            raise ValueError("All learning rates must be positive.")
        if self.lr_decay_type not in {"linear", "exponential", "constant"}:
            raise ValueError(
                "lr_decay_type must be linear, exponential, or constant."
            )
        if self.test_episodes < 0:
            raise ValueError("test_episodes cannot be negative.")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive.")
        if self.reward_smoothing <= 0:
            raise ValueError("reward_smoothing must be positive.")
