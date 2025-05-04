import torch
import torch.nn as nn

class WorldModelMLP(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        # Optional reward prediction head
        # self.fc_r = nn.Linear(hidden_dim, 1)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=-1)
        hidden = torch.relu(self.fc1(za))
        hidden = torch.relu(self.fc2(hidden))
        next_z_pred = self.fc_z(hidden)
        # next_r_pred = self.fc_r(hidden) # Optional
        # return next_z_pred, next_r_pred
        return next_z_pred