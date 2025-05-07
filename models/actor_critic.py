import torch
from torch import nn as nn
from torch.distributions import Normal

from utils import LATENT_DIM, ACTION_DIM


# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Output layer for action means
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # Output layer for action log standard deviations (log_std)
        # Using a learnable parameter per action dimension, not state-dependent initially
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        action_mean = self.fc_mean(x)

        # We use tanh activation on the mean for steering [-1, 1].
        # For gas/brake [0, 1], we could apply sigmoid or (tanh+1)/2 later,
        # but often letting the distribution + clipping handle it works okay.
        # Let's apply tanh to the first dim (steering) explicitly.
        # Keep gas/brake means unbounded for now, will rely on sampling/clipping.
        action_mean = torch.cat([
            torch.tanh(action_mean[:, :1]), # Steering mean bounded [-1, 1]
            action_mean[:, 1:]              # Gas, Brake means unbounded
        ], dim=1)


        action_log_std = self.log_std.expand_as(action_mean) # Same log_std for all states
        action_std = torch.exp(action_log_std)

        # Create the Normal distribution
        dist = Normal(action_mean, action_std)
        return dist


# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Output a single value
        )

    def forward(self, state):
        return self.net(state)
