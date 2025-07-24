import torch
from torch import nn as nn
import math

from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, GRID_SIZE

GRU_HIDDEN_DIM = 512  # Default hidden dimension for GRU layers
GRU_NUM_LAYERS = 3  # Default number of GRU layers


# --- Positional Encoding ---
class PositionalEncoding2D(nn.Module):
    """
    Generates 2D positional encodings for a grid.
    It creates separate encodings for row (y) and column (x) positions
    and concatenates them.
    """

    def __init__(self, d_model: int, grid_size: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be an even number, but got {d_model}")

        self.d_model = d_model
        self.grid_size = grid_size
        d_model_half = d_model // 2

        # 1D positional encoding logic
        def get_1d_pe(max_len, dim):
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
            pe = torch.zeros(max_len, dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe

        pe_y = get_1d_pe(grid_size, d_model_half)  # Row encodings
        pe_x = get_1d_pe(grid_size, d_model_half)  # Column encodings

        # Create the full 2D positional encoding grid
        # Shape: [grid_size * grid_size, d_model]
        pe_2d_grid = torch.zeros(grid_size * grid_size, d_model)
        for r in range(grid_size):
            for c in range(grid_size):
                idx = r * grid_size + c
                pe_2d_grid[idx, :] = torch.cat([pe_y[r], pe_x[c]], dim=-1)

        # Add a zero-vector for the start token at position 0
        pe = torch.cat([torch.zeros(1, d_model), pe_2d_grid], dim=0)

        # Reshape to [1, max_len, d_model] for broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, position_index: int) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, d_model].
            position_index (int): The index of the position (0 for start token, 1 to num_tokens).

        Returns:
            torch.Tensor: Tensor with added positional encoding.
        """
        if position_index >= self.pe.size(1):
            raise IndexError(f"Position index {position_index} is out of range for "
                             f"max_len {self.pe.size(1)} in PositionalEncoding2D.")
        return x + self.pe[:, position_index:position_index + 1, :]


# --- GRU-based World Model (Autoregressive Version) ---
class WorldModelGRU(nn.Module):
    """
    A hierarchical GRU-based world model with a probabilistic temporal state
    and a deterministic spatial generation model.
    """

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            codebook_size: int = VQVAE_NUM_EMBEDDINGS,
            hidden_dim: int = GRU_HIDDEN_DIM,
            grid_size: int = GRID_SIZE,
            num_gru_layers: int = GRU_NUM_LAYERS,
            dropout_rate: float = 0.1
    ):
        """
        Initializes the World Model layers.

        Args:
            latent_dim (int): The dimension of each latent vector from the VQ-VAE.
            action_dim (int): The dimension of the action vector.
            hidden_dim (int): The dimension of the GRU's hidden state for each layer.
            codebook_size (int): The size of the VQ-VAE codebook.
            grid_size (int): The size of the input grid (e.g., 4 for a 4x4 map).
            num_gru_layers (int): Number of stacked GRU layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.codebook_size = codebook_size
        self.num_gru_layers = num_gru_layers
        self.dropout_rate = dropout_rate

        # --- Input Processing & Embeddings ---
        self.dropout = nn.Dropout(dropout_rate)

        # --- Input Processing Layers ---
        # Embed the discrete VQ-VAE tokens and continuous actions.
        self.token_embedding = nn.Embedding(codebook_size, latent_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        # Token projection projects VQ-VAE token embedding to GRU's hidden dimension
        self.token_proj = nn.Linear(latent_dim, hidden_dim)

        # --- Learnable Start Token ---
        # This token is projected to hidden_dim, like other token embeddings.
        # It will be of shape [1, 1, hidden_dim] to allow easy batch expansion.
        self.start_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding2D(hidden_dim, grid_size)

        # --- Hierarchical Recurrent Core ---
        self.temporal_grus = nn.ModuleList()
        self.temporal_grus.append(nn.GRUCell(hidden_dim, hidden_dim))
        for _ in range(1, num_gru_layers):
            self.temporal_grus.append(nn.GRUCell(hidden_dim, hidden_dim))

        self.spatial_grus = nn.ModuleList()
        self.spatial_grus.append(nn.GRUCell(hidden_dim, hidden_dim))
        for _ in range(1, num_gru_layers):
            self.spatial_grus.append(nn.GRUCell(hidden_dim, hidden_dim))

        # --- Prediction Heads ---
        # This head predicts the parameters for the temporal state's distribution
        self.temporal_dist_head = nn.Linear(hidden_dim, hidden_dim * 2)

        self.token_prediction_head = nn.Linear(hidden_dim, codebook_size)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self,
                action: torch.Tensor,
                prev_temporal_hidden: torch.Tensor,
                ground_truth_next_tokens: torch.Tensor = None,
                teacher_forcing_prob: float = 1.0):
        """
        Predicts the next state using a hierarchical and probabilistic GRU structure.

        Args:
            action (torch.Tensor): Action taken, a_t.
            prev_temporal_hidden (torch.Tensor): High-level hidden state from the previous step, h_{t-1}.
            ground_truth_next_tokens (torch.Tensor, optional): Ground truth tokens for s_{t+1}.
            teacher_forcing_prob (float): Probability for scheduled sampling.

        Returns:
            Tuple of (predicted_logits, predicted_reward, predicted_done_logits, final_temporal_hidden)
        """
        batch_size = action.size(0)
        device = action.device

        # --- Deterministic Temporal Update (Outer GRU) ---
        # Update the high-level state based on the action.
        action_embed = self.dropout(self.action_embedding(action))

        current_temporal_input = action_embed
        next_deterministic_hidden_layers = []
        for i in range(self.num_gru_layers):
            h_next = self.temporal_grus[i](current_temporal_input, prev_temporal_hidden[i])
            current_temporal_input = self.dropout(h_next) if i < self.num_gru_layers - 1 else h_next
            next_deterministic_hidden_layers.append(h_next)

        # Stack the layers to form the full deterministic hidden state h_t
        deterministic_temporal_hidden = torch.stack(next_deterministic_hidden_layers)
        last_layer_deterministic_hidden = deterministic_temporal_hidden[-1]

        # --- Stochastic Temporal State Sampling ---
        # Predict distribution parameters from the deterministic state
        mean, log_var = self.temporal_dist_head(last_layer_deterministic_hidden).chunk(2, dim=-1)
        log_var = torch.tanh(log_var)  # Constrain variance for stability
        temporal_distribution = torch.distributions.Normal(mean, torch.exp(0.5 * log_var))
        stochastic_temporal_sample = temporal_distribution.rsample()

        # --- Combine States for Prediction ---
        # The state used for prediction combines deterministic and stochastic parts.
        # This is a crucial step for stable generation.
        prediction_state = self.dropout(stochastic_temporal_sample + last_layer_deterministic_hidden)

        # --- Predict Immediate Outcomes ---
        # Reward and done are functions of this combined abstract state.
        predicted_reward = self.reward_head(prediction_state)
        predicted_done_logits = self.done_head(prediction_state)

        # --- Spatial Generation (Inner GRU) ---
        # Autoregressively generate the 16 tokens for the next state.
        all_token_logits = []

        # Initialize the spatial GRU's hidden state.
        # Condition the generation on both the deterministic and stochastic states.
        spatial_hidden_state = deterministic_temporal_hidden.clone()
        spatial_hidden_state[-1] = prediction_state  # Inject combined state into the last layer

        # Start generation with the learnable start token
        current_token_embed = self.dropout(self.start_token_embed.expand(batch_size, -1, -1).squeeze(1))

        for token_idx in range(self.num_tokens):
            # Add positional encoding to the current token embedding
            current_token_embed_unsqueezed = current_token_embed.unsqueeze(1)
            input_with_pe = self.pos_encoder(current_token_embed_unsqueezed, token_idx)
            current_spatial_input = self.dropout(input_with_pe.squeeze(1))

            next_spatial_hidden_layers = []
            for l_idx in range(self.num_gru_layers):
                h_prev = spatial_hidden_state[l_idx]
                h_next = self.spatial_grus[l_idx](current_spatial_input, h_prev)
                current_spatial_input = self.dropout(h_next) if l_idx < self.num_gru_layers - 1 else h_next
                next_spatial_hidden_layers.append(h_next)

            spatial_hidden_state = torch.stack(next_spatial_hidden_layers)

            # Predict the token from the spatial GRU's state
            last_layer_spatial_hidden = self.dropout(spatial_hidden_state[-1])
            current_logits = self.token_prediction_head(last_layer_spatial_hidden)
            all_token_logits.append(current_logits)

            # Scheduled sampling logic
            if self.training and ground_truth_next_tokens is not None:
                # --- Scheduled Sampling (Training Mode) ---
                # This logic must be graph-compatible for torch.compile.

                # Generate a random tensor for the decision.
                # Shape: [batch_size, 1] to allow per-item decisions.
                prob_tensor = torch.rand(batch_size, 1, device=device)
                use_teacher_force_mask = prob_tensor < teacher_forcing_prob

                # Prepare both branches of the decision.
                teacher_tokens = ground_truth_next_tokens[:, token_idx]
                sampled_tokens = torch.distributions.Categorical(logits=current_logits).sample()

                # Use torch.where to select based on the mask.
                # The mask is broadcasted to match the token tensor shapes.
                next_token_indices = torch.where(use_teacher_force_mask.squeeze(-1), teacher_tokens, sampled_tokens)
            else:
                # --- Autoregressive Sampling (Inference Mode) ---
                # Always sample from the model's own predictions.
                next_token_indices = torch.distributions.Categorical(logits=current_logits).sample()

            # Prepare the next input for the spatial GRU
            current_token_embed = self.dropout(self.token_proj(self.token_embedding(next_token_indices)))

        predicted_latent_logits = torch.stack(all_token_logits, dim=1).view(
            batch_size, self.grid_size, self.grid_size, -1
        )

        # --- Return the hidden state for the next timestep ---
        # The final hidden state that matters for the next time step is the deterministic one.
        final_temporal_hidden = deterministic_temporal_hidden

        return predicted_latent_logits, predicted_reward, predicted_done_logits, final_temporal_hidden

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Helper function to get a zero-initialized hidden state for the temporal GRU."""
        return torch.zeros(self.num_gru_layers, batch_size, self.hidden_dim, device=device)

    def encode_observation(self, observation_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encodes a sequence of real observation tokens into a deterministic hidden state representation.
        This representation is then used as the input to the temporal model to produce a distribution.
        """
        batch_size = observation_tokens.size(0)
        device = observation_tokens.device

        spatial_hidden_state = self.get_initial_hidden_state(batch_size, device)
        current_token_embed = self.start_token_embed.expand(batch_size, -1, -1).squeeze(1)

        for token_idx in range(self.num_tokens):
            current_token_embed_unsqueezed = current_token_embed.unsqueeze(1)
            input_with_pe = self.pos_encoder(current_token_embed_unsqueezed, token_idx)
            current_spatial_input = self.dropout(input_with_pe.squeeze(1))

            next_spatial_hidden_layers = []
            for l_idx in range(self.num_gru_layers):
                h_prev = spatial_hidden_state[l_idx]
                h_next = self.spatial_grus[l_idx](current_spatial_input, h_prev)
                current_spatial_input = self.dropout(h_next) if l_idx < self.num_gru_layers - 1 else h_next
                next_spatial_hidden_layers.append(h_next)

            spatial_hidden_state = torch.stack(next_spatial_hidden_layers)

            if token_idx < self.num_tokens - 1:
                next_token_indices = observation_tokens[:, token_idx + 1]
                current_token_embed = self.token_proj(self.token_embedding(next_token_indices))

        return spatial_hidden_state


# --- Usage Example ---
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    BATCH_SIZE = 32
    LATENT_DIM = 64  # VQVAE embedding dim
    ACTION_DIM_EXAMPLE = 3
    GRU_HIDDEN_DIM_EXAMPLE = 256  # Per layer
    NUM_GRU_LAYERS_EXAMPLE = 2
    DROPOUT_RATE_EXAMPLE = 0.1
    CODEBOOK_SIZE_EXAMPLE = VQVAE_NUM_EMBEDDINGS
    GRID_SIZE_EXAMPLE = GRID_SIZE
    DEVICE_EXAMPLE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"--- Running WorldModelGRU Example with {NUM_GRU_LAYERS_EXAMPLE} GRU layers and dropout {DROPOUT_RATE_EXAMPLE} ---")

    # --- Instantiate the Model ---
    world_model_example = WorldModelGRU(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM_EXAMPLE,
        hidden_dim=GRU_HIDDEN_DIM_EXAMPLE,
        codebook_size=CODEBOOK_SIZE_EXAMPLE,
        grid_size=GRID_SIZE_EXAMPLE,
        num_gru_layers=NUM_GRU_LAYERS_EXAMPLE,
        dropout_rate=DROPOUT_RATE_EXAMPLE
    ).to(DEVICE_EXAMPLE)

    print(f"World Model created on device: {DEVICE_EXAMPLE}")
    print(f"Number of parameters: {sum(p.numel() for p in world_model_example.parameters()):,}")

    # --- Create Dummy Input Data ---
    action_dummy = torch.randn(BATCH_SIZE, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    # Initial hidden state for the temporal GRU (e.g., at the start of an episode)
    initial_hidden_state = world_model_example.get_initial_hidden_state(BATCH_SIZE, DEVICE_EXAMPLE)
    print(f"Initial hidden_state shape: {initial_hidden_state.shape}")

    # Dummy tokens for a real observation s_t
    current_obs_tokens_dummy = torch.randint(0, CODEBOOK_SIZE_EXAMPLE,
                                             (BATCH_SIZE, GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE)).to(DEVICE_EXAMPLE)
    # Ground truth tokens for the next observation s_{t+1} (for teacher forcing)
    ground_truth_next_tokens_dummy = torch.randint(0, CODEBOOK_SIZE_EXAMPLE,
                                                   (BATCH_SIZE, GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE)).to(
        DEVICE_EXAMPLE)

    # --- Encode a real observation to get its hidden state representation ---
    print("\n--- Encoding Observation ---")
    # This would be used to get the hidden state h_t from observation s_t
    # which then becomes prev_temporal_hidden for predicting s_{t+1}
    encoded_hidden_state = world_model_example.encode_observation(current_obs_tokens_dummy)
    print(f"Encoded hidden state shape: {encoded_hidden_state.shape}")

    # --- Perform a Forward Pass (Training with Teacher Forcing) ---
    print("\n--- Running in Training Mode (Teacher Forcing) ---")
    # We use the encoded_hidden_state (h_t) and an action (a_t) to predict the next state (s_{t+1})
    predicted_logits, reward, done_logits, next_hidden_state = world_model_example(
        action=action_dummy,
        prev_temporal_hidden=encoded_hidden_state,
        ground_truth_next_tokens=ground_truth_next_tokens_dummy,
        teacher_forcing_prob=0.75  # Example probability
    )
    print(f"Predicted Logits Shape: {predicted_logits.shape}")  # Should be [B, G, G, Codebook]
    print(f"Predicted Reward Shape:  {reward.shape}")  # Should be [B, 1]
    print(f"Predicted Done Shape:    {done_logits.shape}")  # Should be [B, 1]
    print(f"Next Hidden State Shape: {next_hidden_state.shape}")  # Should be [NumLayers, B, HiddenDim]

    # --- Perform a Forward Pass (Inference without Teacher Forcing) ---
    print("\n--- Running in Inference Mode ---")
    world_model_example.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predicted_logits_inf, _, _, _ = world_model_example(
            action=action_dummy,
            prev_temporal_hidden=encoded_hidden_state,
            ground_truth_next_tokens=None  # No ground truth provided
        )
    print(f"Predicted Logits Shape (Inference): {predicted_logits_inf.shape}")
    world_model_example.train()  # Set model back to training mode
