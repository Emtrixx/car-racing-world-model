import torch
from torch import nn as nn
from torch.distributions import Normal

from src.utils import ACTION_DIM, NUM_STACK
from src.legacy.utils_legacy import LATENT_DIM


# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim=LATENT_DIM * NUM_STACK, action_dim=ACTION_DIM, hidden_dim=256, log_std_min=-20,
                 log_std_max=2):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        # Output layer for action means
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # Output layer for action log standard deviations (state-dependent)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        # Initialize weights and biases for fc_log_std
        # This helps to start with a reasonable initial standard deviation
        torch.nn.init.zeros_(self.fc_log_std.weight)
        # Initial bias of -1.0 means initial std ~ exp(-1) ~ 0.36
        # Adjust as needed based on action scale and desired initial exploration
        torch.nn.init.constant_(self.fc_log_std.bias, -1.0)

    def forward(self, state):
        x = self.net(state)
        action_mean = self.fc_mean(x)

        # Compute state-dependent log standard deviation
        action_log_std_unbounded = self.fc_log_std(x)
        action_log_std = torch.clamp(action_log_std_unbounded, self.log_std_min, self.log_std_max)
        action_std = torch.exp(action_log_std)

        # Create the Normal distribution
        dist = Normal(action_mean, action_std)
        return dist


# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim=LATENT_DIM * NUM_STACK, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Output a single value
        )

    def forward(self, state):
        return self.net(state)
