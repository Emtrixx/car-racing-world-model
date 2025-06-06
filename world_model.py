import torch
from torch import nn as nn
from torch.nn import functional as F

from utils import LATENT_DIM, ACTION_DIM


# --- MLP-based World Model ---
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
    def __init__(self,
                 action_dim,
                 codebook_size,  # K: Size of your VQ-VAE codebook
                 token_embedding_dim,  # Embedding dimension for the single observation token
                 gru_hidden_dim=256,
                 gru_num_layers=1,
                 gru_input_embed_dim=None):  # Optional: dimension after projecting combined obs+action
        super().__init__()
        self.action_dim = action_dim
        self.codebook_size = codebook_size
        self.token_embedding_dim = token_embedding_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers

        # --- Input Processing Layers ---
        # 1. Embedding layer for the single observation token
        self.obs_token_embed = nn.Embedding(codebook_size, token_embedding_dim)

        # The input size to the (optional) projection layer or GRU
        # is the sum of the single token's embedding dim and action dim
        input_to_projection_size = token_embedding_dim + action_dim

        if gru_input_embed_dim is None:
            self.gru_input_size = input_to_projection_size
            self.input_embed_projection = None  # No separate projection layer before GRU
        else:
            self.gru_input_size = gru_input_embed_dim
            # 2. Linear layer to project concatenated (embedded_obs_token + action)
            self.input_embed_projection = nn.Linear(input_to_projection_size, gru_input_embed_dim)

        # 3. GRU layer
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True
        )

        # --- Output Layers (Predicting Next Token Logits) ---
        # The GRU output will be used to predict the logits for the single next observation token.
        self.fc_out_obs_logits = nn.Linear(gru_hidden_dim, codebook_size)
        self.fc_r_pred = nn.Linear(gru_hidden_dim, 1)
        self.fc_d_pred = nn.Linear(gru_hidden_dim, 1)


def forward(self, obs_token_indices_sequence, a_sequence, h_initial=None):
    """
    Args:
        obs_token_indices_sequence (torch.Tensor): Sequence of single observation token indices.
                                        Shape: (batch_size, sequence_length), dtype=torch.long
        a_sequence (torch.Tensor): Actions sequence.
                                   Shape: (batch_size, sequence_length, action_dim)
        h_initial (torch.Tensor, optional): Initial hidden state for GRU.
    Returns:
        next_obs_logits_sequence (torch.Tensor): Predicted logits for the next single observation token.
                                      Shape: (batch_size, sequence_length, codebook_size)
        next_r_pred_sequence (torch.Tensor): Predicted rewards.
        next_d_pred_logits (torch.Tensor): Predicted done logits.
        h_final (torch.Tensor): Final hidden state of GRU.
    """
    # Ensure actions are float and on the same device
    if a_sequence.dtype != torch.float32:
        a_sequence = a_sequence.float()
    a_sequence = a_sequence.to(obs_token_indices_sequence.device)

    # Embed observation tokens
    # Input: (batch, seq)
    # Output: (batch, seq, token_embedding_dim)
    embedded_obs_tokens = self.obs_token_embed(obs_token_indices_sequence)

    # Concatenate embedded observation tokens with actions
    # Output: (batch, seq, token_embedding_dim + action_dim)
    obs_action_combined = torch.cat([embedded_obs_tokens, a_sequence], dim=-1)

    # Optional: Project to gru_input_embed_dim
    if self.input_embed_projection:
        projected_input = F.relu(self.input_embed_projection(obs_action_combined))
    else:
        projected_input = obs_action_combined

    # Pass through GRU
    gru_output, h_final = self.gru(projected_input, h_initial)

    # Predict next observation token logits, reward, and done
    # Output shape: (batch_size, sequence_length, codebook_size)
    next_obs_logits_sequence = self.fc_out_obs_logits(gru_output)

    next_r_pred_sequence = self.fc_r_pred(gru_output)
    next_d_pred_logits = self.fc_d_pred(gru_output)

    return next_obs_logits_sequence, next_r_pred_sequence, next_d_pred_logits, h_final


def step(self, obs_token_idx_t, a_t, h_prev):
    """
    Performs a single step prediction.
    Args:
        obs_token_idx_t (torch.Tensor): Current single observation token index.
                                  Shape: (batch_size,), dtype=torch.long
        a_t (torch.Tensor): Current action. Shape: (batch_size, action_dim)
        h_prev (torch.Tensor): Previous hidden state.
    Returns:
        next_obs_logits (torch.Tensor): Predicted logits for the next single obs token.
                             Shape: (batch_size, codebook_size)
        next_r_pred (torch.Tensor): Predicted reward.
        next_d_pred_logit (torch.Tensor): Predicted done logit.
        h_next (torch.Tensor): Next hidden state.
    """
    # Reshape inputs to (batch_size, sequence_length=1, features)
    # obs_token_idx_t shape: (batch_size,) -> (batch_size, 1)
    obs_token_idx_t_seq = obs_token_idx_t.unsqueeze(1)
    # a_t shape: (batch_size, action_dim) -> (batch_size, 1, action_dim)
    a_t_seq = a_t.unsqueeze(1)

    next_obs_logits_seq, next_r_pred_seq, next_d_logits_seq, h_next = \
        self.forward(obs_token_idx_t_seq, a_t_seq, h_initial=h_prev)

    # Squeeze the sequence_length=1 dimension
    # Output shape: (batch_size, codebook_size)
    next_obs_logits = next_obs_logits_seq.squeeze(1)
    next_r_pred = next_r_pred_seq.squeeze(1)
    next_d_pred_logit = next_d_logits_seq.squeeze(1)

    return next_obs_logits, next_r_pred, next_d_pred_logit, h_next
