import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import LATENT_DIM, ACTION_DIM


# --- MLO-based World Model ---
class WorldModelMLP(nn.Module):
    # Use WM_HIDDEN_DIM from utils or define a default
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_z_pred = nn.Linear(hidden_dim, latent_dim)
        # Optional heads can be added later
        # self.fc_r_pred = nn.Linear(hidden_dim, 1)

    def forward(self, z, a):
        # Ensure a is float tensor and on same device as z
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32, device=z.device)
        elif a.dtype != torch.float32:
            a = a.float()
        a = a.to(z.device)

        # Handle batch dimension if missing (e.g., during single step prediction)
        if z.ndim == 1: z = z.unsqueeze(0)
        if a.ndim == 1: a = a.unsqueeze(0)

        za = torch.cat([z, a], dim=-1)
        hidden = F.relu(self.fc1(za))
        hidden = F.relu(self.fc2(hidden))
        next_z_pred = self.fc_z_pred(hidden)
        return next_z_pred


# --- GRU-based World Model ---
class WorldModelGRU(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                 gru_hidden_dim=256, gru_num_layers=1, gru_input_embed_dim=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers

        if gru_input_embed_dim is None:
            self.gru_input_size = latent_dim + action_dim
            self.input_embed = None
        else:
            self.gru_input_size = gru_input_embed_dim
            self.input_embed = nn.Linear(latent_dim + action_dim, gru_input_embed_dim)

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True # Expects input: (batch, seq, feature)
        )
        # Output layer projects GRU hidden state to the predicted next latent state
        self.fc_out = nn.Linear(gru_hidden_dim, latent_dim)

    def forward(self, z_sequence, a_sequence, h_initial=None):
        """
        Args:
            z_sequence (torch.Tensor): Latent states sequence.
                                       Shape: (batch_size, sequence_length, latent_dim)
            a_sequence (torch.Tensor): Actions sequence.
                                       Shape: (batch_size, sequence_length, action_dim)
            h_initial (torch.Tensor, optional): Initial hidden state for GRU.
                                       Shape: (gru_num_layers, batch_size, gru_hidden_dim)
                                       Defaults to None (zeros).
        Returns:
            next_z_pred_sequence (torch.Tensor): Predicted next latent states.
                                       Shape: (batch_size, sequence_length, latent_dim)
            h_final (torch.Tensor): Final hidden state of GRU.
                                       Shape: (gru_num_layers, batch_size, gru_hidden_dim)
        """
        # Ensure actions are float and on the same device
        if a_sequence.dtype != torch.float32:
            a_sequence = a_sequence.float()
        a_sequence = a_sequence.to(z_sequence.device)

        # Concatenate z and a along the feature dimension
        za_sequence = torch.cat([z_sequence, a_sequence], dim=-1)

        if self.input_embed:
            za_sequence = F.relu(self.input_embed(za_sequence))

        # Pass through GRU
        # gru_output shape: (batch_size, sequence_length, gru_hidden_dim)
        # h_final shape: (gru_num_layers, batch_size, gru_hidden_dim)
        gru_output, h_final = self.gru(za_sequence, h_initial)

        # Predict next latent state from GRU output
        next_z_pred_sequence = self.fc_out(gru_output)

        return next_z_pred_sequence, h_final

    def step(self, z_t, a_t, h_prev):
        """
        Performs a single step prediction, useful for dreaming/generation.
        Args:
            z_t (torch.Tensor): Current latent state. Shape: (batch_size, latent_dim)
            a_t (torch.Tensor): Current action. Shape: (batch_size, action_dim)
            h_prev (torch.Tensor): Previous hidden state. Shape: (gru_num_layers, batch_size, gru_hidden_dim)
        Returns:
            next_z_pred (torch.Tensor): Predicted next latent state. Shape: (batch_size, latent_dim)
            h_next (torch.Tensor): Next hidden state. Shape: (gru_num_layers, batch_size, gru_hidden_dim)
        """
        # Reshape inputs to (batch_size, sequence_length=1, features)
        z_t_seq = z_t.unsqueeze(1)
        a_t_seq = a_t.unsqueeze(1)

        next_z_pred_seq, h_next = self.forward(z_t_seq, a_t_seq, h_initial=h_prev)
        next_z_pred = next_z_pred_seq.squeeze(1) # Remove sequence_length dim

        return next_z_pred, h_next
