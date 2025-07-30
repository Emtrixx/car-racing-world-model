from typing import Tuple

import torch
from torch import nn as nn

from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, GRID_SIZE

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
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2)  # mu and log_var for the stochastic state
        )

        # --- Prediction Heads ---
        # Decodes the (deterministic + stochastic) state into predictions.
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),  # deterministic + stochastic state
            nn.ReLU(),
            nn.Linear(d_model, self.num_tokens * codebook_size)
        )
        self.reward_head = nn.Linear(d_model + d_model, 1)
        self.done_head = nn.Linear(d_model + d_model, 1)

    def encode_observation(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Encodes a grid of tokens into a single embedding vector."""
        batch_size = obs_tokens.size(0)
        # Embed and flatten all tokens
        embedded_tokens = self.token_embedding(obs_tokens)  # [B, num_tokens, d_model]
        flat_embedded_tokens = embedded_tokens.view(batch_size, -1)  # [B, num_tokens * d_model]
        # Encode into a single vector
        obs_embedding = self.encoder(flat_embedded_tokens)
        return obs_embedding

    def forward(
            self,
            prev_obs_tokens: torch.Tensor,
            prev_action: torch.Tensor,
            prev_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a single-step rollout (imagine one timestep).

        Args:
            prev_obs_tokens (torch.Tensor): Previous observation tokens, s_{t-1}.
            prev_action (torch.Tensor): Previous action, a_{t-1}.
            prev_hidden_state (torch.Tensor): Previous GRU hidden state.

        Returns:
            A tuple containing:
            - Predicted next token logits.
            - Predicted reward.
            - Predicted done logits.
            - The new deterministic hidden state for the next step.
            - The sampled stochastic state used for prediction.
        """
        # Encode the previous observation tokens into a fixed-size embedding
        obs_embedding = self.encode_observation(prev_obs_tokens)

        # Embed the action
        action_embedding = self.action_embedding(prev_action)

        # Update the deterministic hidden state
        gru_input = torch.cat([obs_embedding, action_embedding], dim=1).unsqueeze(1)
        deterministic_state, new_hidden_state = self.recurrent_core(gru_input, prev_hidden_state)
        deterministic_state = deterministic_state.squeeze(1)  # [B, d_model]

        # Predict the stochastic state from the deterministic one
        mean, log_var = self.stochastic_predictor(deterministic_state).chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        stochastic_distribution = torch.distributions.Normal(mean, std)
        stochastic_state = stochastic_distribution.rsample()

        # Make predictions using the combined state
        combined_state = torch.cat([deterministic_state, stochastic_state], dim=1)

        # Predict token logits
        predicted_token_logits_flat = self.decoder_head(combined_state)
        predicted_token_logits = predicted_token_logits_flat.view(
            -1, self.num_tokens, self.codebook_size
        )

        # Predict reward and done
        predicted_reward = self.reward_head(combined_state)
        predicted_done = self.done_head(combined_state)

        return (
            predicted_token_logits,
            predicted_reward,
            predicted_done,
            new_hidden_state,
            stochastic_state
        )

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes the GRU hidden state."""
        return torch.zeros(self.recurrent_core.num_layers, batch_size, self.d_model, device=device)


# --- Usage Example ---
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    BATCH_SIZE = 32
    ACTION_DIM_EXAMPLE = 3
    D_MODEL_EXAMPLE = 256  # d_model for the model
    NUM_GRU_LAYERS_EXAMPLE = 2
    CODEBOOK_SIZE_EXAMPLE = VQVAE_NUM_EMBEDDINGS
    GRID_SIZE_EXAMPLE = GRID_SIZE
    DEVICE_EXAMPLE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"--- Running WorldModelGRU Example with {NUM_GRU_LAYERS_EXAMPLE} GRU layers and d_model {D_MODEL_EXAMPLE} ---")

    # --- Instantiate the Model ---
    world_model_example = WorldModelGRU(
        action_dim=ACTION_DIM_EXAMPLE,
        d_model=D_MODEL_EXAMPLE,
        gru_num_layers=NUM_GRU_LAYERS_EXAMPLE,
        codebook_size=CODEBOOK_SIZE_EXAMPLE,
        grid_size=GRID_SIZE_EXAMPLE,
    ).to(DEVICE_EXAMPLE)

    print(f"World Model created on device: {DEVICE_EXAMPLE}")
    print(f"Number of parameters: {sum(p.numel() for p in world_model_example.parameters()):,}")

    # --- Create Dummy Input Data ---
    # Previous action a_{t-1}
    prev_action_dummy = torch.randn(BATCH_SIZE, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    # Previous observation s_{t-1}
    prev_obs_tokens_dummy = torch.randint(
        0, CODEBOOK_SIZE_EXAMPLE,
        (BATCH_SIZE, GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE)
    ).to(DEVICE_EXAMPLE)
    # Initial hidden state for the GRU h_{t-1}
    initial_hidden_state = world_model_example.get_initial_hidden_state(BATCH_SIZE, DEVICE_EXAMPLE)

    print(f"\n--- Input Shapes ---")
    print(f"Previous Action Shape:      {prev_action_dummy.shape}")
    print(f"Previous Obs Tokens Shape:  {prev_obs_tokens_dummy.shape}")
    print(f"Initial GRU Hidden Shape:   {initial_hidden_state.shape}")

    # --- Test encode_observation separately ---
    print("\n--- Testing Observation Encoding ---")
    obs_embedding = world_model_example.encode_observation(prev_obs_tokens_dummy)
    print(f"Encoded observation embedding shape: {obs_embedding.shape}")  # Should be [B, d_model]

    # --- Perform a Forward Pass (a single rollout step) ---
    print("\n--- Running a single forward pass ---")
    # From s_{t-1}, a_{t-1}, h_{t-1}, predict s_t, r_t, d_t and get new state h_t
    (
        predicted_token_logits,
        predicted_reward,
        predicted_done,
        new_hidden_state,
        stochastic_state
    ) = world_model_example(
        prev_obs_tokens=prev_obs_tokens_dummy,
        prev_action=prev_action_dummy,
        prev_hidden_state=initial_hidden_state
    )
    print(f"Predicted Token Logits Shape: {predicted_token_logits.shape}")  # Should be [B, num_tokens, Codebook]
    print(f"Predicted Reward Shape:       {predicted_reward.shape}")  # Should be [B, 1]
    print(f"Predicted Done Shape:         {predicted_done.shape}")  # Should be [B, 1]
    print(f"New GRU Hidden State Shape:   {new_hidden_state.shape}")  # Should be [NumLayers, B, d_model]
    print(f"Sampled Stochastic State Shape: {stochastic_state.shape}")  # Should be [B, d_model]

    # --- Perform a Forward Pass in Eval Mode ---
    print("\n--- Running in Eval Mode ---")
    world_model_example.eval()
    with torch.no_grad():
        (
            predicted_token_logits_inf,
            predicted_reward_inf,
            predicted_done_inf,
            new_hidden_state_inf,
            stochastic_state_inf
        ) = world_model_example(
            prev_obs_tokens=prev_obs_tokens_dummy,
            prev_action=prev_action_dummy,
            prev_hidden_state=initial_hidden_state
        )
    print(f"Predicted Token Logits Shape (Inference): {predicted_token_logits_inf.shape}")
    world_model_example.train()  # Set back to train mode
