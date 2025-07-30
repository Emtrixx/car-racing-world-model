from typing import Tuple

import torch
from torch import nn as nn

from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, GRID_SIZE
from src.vq_conv_vae import VQVAE_EMBEDDING_DIM

GRU_NUM_LAYERS = 3  # Default number of GRU layers
D_MODEL = 1024


# --- GRU-based World Model (Autoregressive Version) ---
class WorldModelGRU(nn.Module):
    """
    A GRU-based world model inspired by Dreamer-family architectures,
    adapted for a discrete, tokenized observation space. It uses a recurrent
    state to predict the next latent image, reward, and termination signal.
    """

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            d_model: int = D_MODEL,
            gru_num_layers: int = 3,
            codebook_size: int = VQVAE_NUM_EMBEDDINGS,
            grid_size: int = GRID_SIZE,
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.codebook_size = codebook_size

        # --- State Encoding ---
        # A network to process the full grid of tokens into a single state embedding.
        # This replaces the need for a "spatial GRU".
        self.token_embedding = nn.Embedding(codebook_size, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim * self.num_tokens, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

        # --- Recurrent Core (RSSM-style) ---
        # Input to GRU: Encoded observation + action embedding
        self.action_embedding = nn.Linear(action_dim, d_model)
        self.recurrent_core = nn.GRU(
            input_size=d_model + d_model,  # action_embed + obs_embed
            hidden_size=d_model,
            num_layers=gru_num_layers,
            batch_first=True
        )

        # --- Latent State Prediction ---
        # Predicts the next latent state distribution from the recurrent hidden state.
        self.stochastic_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)  # mu and log_var for the stochastic state
        )

        # --- Prediction Heads ---
        # Decodes the (deterministic + stochastic) state into predictions.
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),  # deterministic + stochastic state
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.num_tokens * codebook_size)
        )
        self.reward_head = nn.Linear(d_model + d_model, 1)
        self.done_head = nn.Linear(d_model + d_model, 1)

    def encode_observation(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of token grids into embedding vectors."""
        batch_size, seq_len, num_tokens = obs_tokens.shape
        obs_tokens = obs_tokens.view(batch_size * seq_len, num_tokens)
        embedded_tokens = self.token_embedding(obs_tokens)
        flat_embedded = embedded_tokens.view(batch_size * seq_len, -1)
        obs_embedding = self.encoder(flat_embedded)
        return obs_embedding.view(batch_size, seq_len, self.d_model)

    def forward(
            self,
            obs_tokens: torch.Tensor,
            actions: torch.Tensor,
            initial_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "torch.distributions.Distribution"]:
        """
        Processes a sequence of observations and actions.

        Args:
            obs_tokens (torch.Tensor): Sequence of observation tokens [B, T, N].
            actions (torch.Tensor): Sequence of actions [B, T, A].
            initial_hidden_state (torch.Tensor): Initial GRU hidden state.

        Returns:
            Tuple of predicted sequences and final states.
        """
        obs_embeddings = self.encode_observation(obs_tokens)
        action_embeddings = self.action_embedding(actions)

        gru_input = torch.cat([obs_embeddings, action_embeddings], dim=-1)
        deterministic_states, final_hidden_state = self.recurrent_core(gru_input, initial_hidden_state)

        mean, log_var = self.stochastic_predictor(deterministic_states).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        stochastic_dist = torch.distributions.Normal(mean, std)
        stochastic_states = stochastic_dist.rsample()

        combined_states = torch.cat([deterministic_states, stochastic_states], dim=-1)

        predicted_logits_flat = self.decoder_head(combined_states)
        predicted_token_logits = predicted_logits_flat.view(
            obs_tokens.size(0), obs_tokens.size(1), self.num_tokens, self.codebook_size
        )
        predicted_rewards = self.reward_head(combined_states)
        predicted_dones = self.done_head(combined_states)

        return (
            predicted_token_logits,
            predicted_rewards,
            predicted_dones,
            final_hidden_state,
            stochastic_dist
        )

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes the GRU hidden state."""
        return torch.zeros(self.recurrent_core.num_layers, batch_size, self.d_model, device=device)


# --- Usage Example ---
if __name__ == '__main__':
    BATCH_SIZE = 32
    SEQ_LENGTH = 10
    ACTION_DIM_EXAMPLE = 3
    D_MODEL_EXAMPLE = 256
    NUM_GRU_LAYERS_EXAMPLE = 2
    CODEBOOK_SIZE_EXAMPLE = VQVAE_NUM_EMBEDDINGS
    GRID_SIZE_EXAMPLE = GRID_SIZE
    DEVICE_EXAMPLE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_model = WorldModelGRU(
        latent_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM_EXAMPLE,
        d_model=D_MODEL_EXAMPLE,
        gru_num_layers=NUM_GRU_LAYERS_EXAMPLE,
        codebook_size=CODEBOOK_SIZE_EXAMPLE,
        grid_size=GRID_SIZE_EXAMPLE,
    ).to(DEVICE_EXAMPLE)

    obs_tokens_dummy = torch.randint(
        0, CODEBOOK_SIZE_EXAMPLE,
        (BATCH_SIZE, SEQ_LENGTH, GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE)
    ).to(DEVICE_EXAMPLE)
    actions_dummy = torch.randn(BATCH_SIZE, SEQ_LENGTH, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    initial_hidden_state = world_model.get_initial_hidden_state(BATCH_SIZE, DEVICE_EXAMPLE)

    logits, rewards, dones, _, _ = world_model(obs_tokens_dummy, actions_dummy, initial_hidden_state)

    print(f"Logits shape: {logits.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Dones shape: {dones.shape}")
