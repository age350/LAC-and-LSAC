"""Actor, twin Q critics, and Lyapunov critic used by LSAC."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Normal


def build_mlp(layer_sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for index in range(len(layer_sizes) - 1):
        layer_activation = (
            activation if index < len(layer_sizes) - 2 else output_activation
        )
        layers.append(nn.Linear(layer_sizes[index], layer_sizes[index + 1]))
        layers.append(layer_activation())
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Stochastic Gaussian actor from Soft Actor-Critic."""

    def __init__(self, obs_dim, action_space, hidden_sizes):
        super().__init__()
        act_dim = action_space.shape[0]
        self.net = build_mlp(
            [obs_dim, *hidden_sizes],
            activation=nn.ReLU,
            output_activation=nn.ReLU,
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs, deterministic=False, with_logprob=True):
        features = self.net(obs)
        mu = self.mu_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), -20.0, 2.0)
        distribution = Normal(mu, log_std.exp())
        raw_action = mu if deterministic else distribution.rsample()

        if with_logprob:
            log_prob = distribution.log_prob(raw_action).sum(dim=-1)
            correction = 2.0 * (
                np.log(2.0) - raw_action - functional.softplus(-2.0 * raw_action)
            )
            log_prob -= correction.sum(dim=-1)
            log_prob -= torch.log(self.action_scale).sum()
        else:
            log_prob = None

        action = torch.tanh(raw_action)
        action = action * self.action_scale + self.action_bias
        return action, log_prob


class QCritic(nn.Module):
    """One SAC state-action value network."""

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.net = build_mlp(
            [obs_dim + act_dim, *hidden_sizes, 1],
            activation=nn.ReLU,
            output_activation=nn.Identity,
        )

    def forward(self, obs, action):
        value = self.net(torch.cat((obs, action), dim=-1))
        return value.squeeze(-1)


class LyapunovCritic(nn.Module):
    """Non-negative Lyapunov critic represented by a sum of squares."""

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.feature_net = build_mlp(
            [obs_dim + act_dim, *hidden_sizes],
            activation=nn.ReLU,
            output_activation=nn.ReLU,
        )

    def forward(self, obs, action):
        features = self.feature_net(torch.cat((obs, action), dim=-1))
        return torch.square(features).sum(dim=-1)


class LSACNetwork(nn.Module):
    """SAC actor and twin Q critics extended with a Lyapunov critic."""

    def __init__(self, observation_space, action_space, hidden_sizes):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.actor = SquashedGaussianActor(obs_dim, action_space, hidden_sizes)
        self.q1 = QCritic(obs_dim, act_dim, hidden_sizes)
        self.q2 = QCritic(obs_dim, act_dim, hidden_sizes)
        self.lyapunov = LyapunovCritic(obs_dim, act_dim, hidden_sizes)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action, _ = self.actor(
                obs, deterministic=deterministic, with_logprob=False
            )
        return action.cpu().numpy()

    def target_copy(self):
        target = copy.deepcopy(self)
        for parameter in target.parameters():
            parameter.requires_grad = False
        return target
