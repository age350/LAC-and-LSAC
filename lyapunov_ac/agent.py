"""Core Lyapunov Actor-Critic update."""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import networks


class LACAgent(nn.Module):
    """Single Lyapunov critic, infinite-horizon LAC."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        gamma=0.99,
        polyak=0.995,
        alpha=0.99,
        alpha3=0.2,
        labda=0.99,
        actor_lr=1e-4,
        critic_lr=3e-4,
        alpha_lr=1e-4,
        labda_lr=3e-4,
        device="cpu",
    ):
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.polyak = polyak
        self.alpha3 = alpha3
        self.target_entropy = -float(np.prod(action_space.shape))

        self.network = networks.LyapunovActorCritic(
            observation_space, action_space, hidden_sizes
        ).to(self.device)
        self.target_network = self.network.target_copy().to(self.device)

        self.log_alpha = nn.Parameter(
            torch.tensor(
                np.log(max(alpha, 1e-37)),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.log_labda = nn.Parameter(
            torch.tensor(
                np.log(max(labda, 1e-37)),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.actor_optimizer = Adam(
            self.network.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = Adam(
            self.network.critic.parameters(), lr=critic_lr
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        self.labda_optimizer = Adam([self.log_labda], lr=labda_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().clamp(min=0.0)

    @property
    def labda(self):
        return self.log_labda.exp().clamp(min=0.0, max=1.0)

    def get_action(self, obs, deterministic=False):
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        )
        return self.network.act(obs_tensor, deterministic=deterministic)

    def set_learning_rates(
        self, actor_lr, critic_lr, alpha_lr, labda_lr
    ):
        """Set optimizer learning rates during scheduled decay."""
        learning_rates = (
            (self.actor_optimizer, actor_lr),
            (self.critic_optimizer, critic_lr),
            (self.alpha_optimizer, alpha_lr),
            (self.labda_optimizer, labda_lr),
        )
        for optimizer, learning_rate in learning_rates:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

    def update(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        cost = batch["cost"]
        obs_next = batch["obs_next"]
        done = batch["done"]

        with torch.no_grad():
            target_action, _ = self.target_network.actor(obs_next)
            target_l = self.target_network.critic(obs_next, target_action)
            l_backup = cost + self.gamma * (1.0 - done) * target_l

        current_l = self.network.critic(obs, action)
        critic_loss = 0.5 * torch.mean(
            torch.square(current_l - l_backup)
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for parameter in self.network.critic.parameters():
            parameter.requires_grad = False

        _, log_prob = self.network.actor(obs)
        next_action, _ = self.network.actor(obs_next)
        next_l = self.network.critic(obs_next, next_action)
        l_delta = torch.mean(
            next_l - current_l.detach() + self.alpha3 * cost
        )
        actor_loss = (
            self.labda.detach() * l_delta
            + self.alpha.detach() * torch.mean(log_prob)
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for parameter in self.network.critic.parameters():
            parameter.requires_grad = True

        alpha_loss = -torch.mean(
            self.alpha * (log_prob.detach() + self.target_entropy)
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        labda_loss = -(self.log_labda * l_delta.detach())
        self.labda_optimizer.zero_grad()
        labda_loss.backward()
        self.labda_optimizer.step()

        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.network.parameters(), self.target_network.parameters()
            ):
                target_parameter.data.mul_(self.polyak)
                target_parameter.data.add_(
                    (1.0 - self.polyak) * parameter.data
                )

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "labda_loss": float(labda_loss.detach().cpu()),
            "l_delta": float(l_delta.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "labda": float(self.labda.detach().cpu()),
            "entropy": float(-log_prob.detach().mean().cpu()),
        }

    def save(self, path):
        torch.save(
            {
                "network": self.network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "log_labda": self.log_labda.detach().cpu(),
            },
            path,
        )

    def load(self, path):
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )
        self.log_alpha.data.copy_(
            checkpoint["log_alpha"].to(self.device)
        )
        self.log_labda.data.copy_(
            checkpoint["log_labda"].to(self.device)
        )
