"""Lyapunov Soft Actor-Critic: SAC plus a Lyapunov stability constraint."""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import networks


class LSACAgent(nn.Module):
    """Twin-Q Soft Actor-Critic extended with a Lyapunov critic."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        alpha3=0.2,
        labda=0.1,
        actor_lr=3e-4,
        q_lr=3e-4,
        lyapunov_lr=3e-4,
        alpha_lr=3e-4,
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

        self.network = networks.LSACNetwork(
            observation_space, action_space, hidden_sizes
        ).to(self.device)
        self.target_network = self.network.target_copy().to(self.device)

        self.log_alpha = nn.Parameter(
            torch.tensor(
                np.log(max(alpha, 1e-8)),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.log_labda = nn.Parameter(
            torch.tensor(
                np.log(max(labda, 1e-8)),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.actor_optimizer = Adam(self.network.actor.parameters(), lr=actor_lr)
        self.q_optimizer = Adam(
            list(self.network.q1.parameters()) + list(self.network.q2.parameters()),
            lr=q_lr,
        )
        self.lyapunov_optimizer = Adam(
            self.network.lyapunov.parameters(), lr=lyapunov_lr
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        self.labda_optimizer = Adam([self.log_labda], lr=labda_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def labda(self):
        return self.log_labda.exp().clamp(max=1.0)

    def get_action(self, obs, deterministic=False):
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        )
        return self.network.act(obs_tensor, deterministic=deterministic)

    def update(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        cost = batch["cost"]
        obs_next = batch["obs_next"]
        done = batch["done"]

        # Standard SAC twin-Q Bellman target.
        with torch.no_grad():
            next_action, next_log_prob = self.network.actor(obs_next)
            target_q1 = self.target_network.q1(obs_next, next_action)
            target_q2 = self.target_network.q2(obs_next, next_action)
            target_q = torch.minimum(target_q1, target_q2)
            q_backup = reward + self.gamma * (1.0 - done) * (
                target_q - self.alpha.detach() * next_log_prob
            )

            target_l_action, _ = self.target_network.actor(obs_next)
            target_l = self.target_network.lyapunov(obs_next, target_l_action)
            l_backup = cost + self.gamma * (1.0 - done) * target_l

        q1 = self.network.q1(obs, action)
        q2 = self.network.q2(obs, action)
        q1_loss = 0.5 * torch.mean(torch.square(q1 - q_backup))
        q2_loss = 0.5 * torch.mean(torch.square(q2 - q_backup))
        q_loss = q1_loss + q2_loss
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        current_l = self.network.lyapunov(obs, action)
        lyapunov_loss = 0.5 * torch.mean(torch.square(current_l - l_backup))
        self.lyapunov_optimizer.zero_grad()
        lyapunov_loss.backward()
        self.lyapunov_optimizer.step()

        frozen_modules = [
            self.network.q1,
            self.network.q2,
            self.network.lyapunov,
        ]
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

        policy_action, log_prob = self.network.actor(obs)
        q1_policy = self.network.q1(obs, policy_action)
        q2_policy = self.network.q2(obs, policy_action)
        min_q_policy = torch.minimum(q1_policy, q2_policy)
        sac_actor_loss = torch.mean(
            self.alpha.detach() * log_prob - min_q_policy
        )

        next_policy_action, _ = self.network.actor(obs_next)
        next_l = self.network.lyapunov(obs_next, next_policy_action)
        l_delta = torch.mean(
            next_l - current_l.detach() + self.alpha3 * cost
        )
        actor_loss = sac_actor_loss + self.labda.detach() * l_delta

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

        alpha_loss = -torch.mean(
            self.log_alpha * (log_prob.detach() + self.target_entropy)
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        labda_loss = -(self.log_labda * l_delta.detach())
        self.labda_optimizer.zero_grad()
        labda_loss.backward()
        self.labda_optimizer.step()
        with torch.no_grad():
            self.log_labda.clamp_(min=np.log(1e-8), max=0.0)

        self._update_targets()

        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "lyapunov_loss": float(lyapunov_loss.detach().cpu()),
            "sac_actor_loss": float(sac_actor_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "labda_loss": float(labda_loss.detach().cpu()),
            "l_delta": float(l_delta.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "labda": float(self.labda.detach().cpu()),
            "entropy": float(-log_prob.detach().mean().cpu()),
            "q_value": float(min_q_policy.detach().mean().cpu()),
            "lyapunov_value": float(current_l.detach().mean().cpu()),
        }

    def _update_targets(self):
        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.network.parameters(), self.target_network.parameters()
            ):
                target_parameter.data.mul_(self.polyak)
                target_parameter.data.add_((1.0 - self.polyak) * parameter.data)

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
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.log_labda.data.copy_(checkpoint["log_labda"].to(self.device))

