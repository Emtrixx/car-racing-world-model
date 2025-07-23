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
    An autoregressive GRU-based world model.

    This model performs two main steps:
    1. A 'transition' step where the action updates the hidden state, which is then
       used to predict the immediate reward and done flag.
    2. An 'autoregressive generation' step where it predicts the latent
       feature map one token at a time as recurrent steps.
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

        # --- Dropout Layer ---
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
        self.pos_encoder = PositionalEncoding2D(hidden_dim, self.grid_size)
        # --- Recurrent Core ---
        self.grus = nn.ModuleList()
        # First GRU layer takes the projected action or token embedding as input
        self.grus.append(nn.GRUCell(hidden_dim, hidden_dim))
        # Subsequent GRU layers take the output of the previous GRU layer as input
        for _ in range(1, num_gru_layers):
            self.grus.append(nn.GRUCell(hidden_dim, hidden_dim))

        # --- Output Prediction Heads ---
        # Prediction heads operate on the output of the last GRU layer.
        self.next_latent_head = nn.Linear(hidden_dim, codebook_size)
        # Predicts the scalar reward.
        self.reward_head = nn.Linear(hidden_dim, 1)
        # Predicts the 'done' logit.
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, action: torch.Tensor, prev_hidden_state: torch.Tensor,
                ground_truth_tokens: torch.Tensor = None,
                teacher_forcing_prob: float = 1.0):
        """
        Performs a single step of the world model prediction autoregressively.

        Args:
            action (torch.Tensor): The action taken. Shape: [batch_size, action_dim]
            prev_hidden_state (torch.Tensor): The previous hidden states of the GRUs.
                                              Shape: [num_gru_layers, batch_size, hidden_dim]
            ground_truth_tokens (torch.Tensor, optional): The ground truth tokens of the
                next state for teacher forcing. Shape: [batch_size, 16].
                If None, the model uses its own predictions (inference).
            teacher_forcing_prob (float): The probability of using teacher forcing for each token.
                                          Defaults to 1.0 (always use teacher forcing if available).

        Returns:
            Tuple containing:
            - predicted_latent_logits (torch.Tensor): Logits for the next latent grid.
              Shape: [batch_size, grid_size, grid_size, codebook_size]
            - predicted_reward (torch.Tensor): The predicted scalar reward.
            - predicted_done_logits (torch.Tensor): The predicted done logit.
            - final_hidden_state (torch.Tensor): The final GRU hidden state after generation.
        """
        batch_size = action.size(0)
        device = action.device

        # --- Transition Step ---
        # Use the action to update the hidden state. This new state summarizes the transition.
        action_embed_raw = self.action_embedding(action)
        action_embed = self.dropout(action_embed_raw)  # Dropout after action embedding

        current_input = action_embed
        prior_hidden_layers = []
        for i in range(self.num_gru_layers):
            h_next_layer = self.grus[i](current_input, prev_hidden_state[i])
            if i < self.num_gru_layers - 1:  # Apply dropout between GRU layers
                current_input = self.dropout(h_next_layer)
            else:  # No dropout after the last GRU layer's output if it goes to heads
                current_input = h_next_layer
            prior_hidden_layers.append(h_next_layer)  # Store original output for hidden state stack

        prior_hidden_state = torch.stack(prior_hidden_layers)

        # --- Update Step (Correction using Observation) ---
        all_reconstruction_logits = []
        posterior_hidden_state = prior_hidden_state

        # --- Autoregressive Generation with Start Token and Positional Encoding ---
        current_pos_idx = 0

        # Prepare the start token: expand to batch size, add positional encoding, apply dropout
        # self.start_token_embed is [1, 1, hidden_dim]
        # expanded_start_token is [batch_size, 1, hidden_dim]
        expanded_start_token = self.start_token_embed.expand(batch_size, -1, -1)
        start_token_with_pe = self.pos_encoder(expanded_start_token, current_pos_idx)
        # Input to GRU should be [batch_size, hidden_dim]
        prev_token_embed_projected = self.dropout(start_token_with_pe.squeeze(1))
        current_pos_idx += 1

        for token_idx in range(self.num_tokens):  # Loop to generate self.num_tokens
            # The input to the GRU is the previous ground-truth token from the observation
            current_gru_input_for_stack = prev_token_embed_projected
            next_hidden_layers = []

            for l_idx in range(self.num_gru_layers):
                h_prev = posterior_hidden_state[l_idx]
                h_next = self.grus[l_idx](current_gru_input_for_stack, h_prev)
                current_gru_input_for_stack = self.dropout(h_next) if l_idx < self.num_gru_layers - 1 else h_next
                next_hidden_layers.append(h_next)

            posterior_hidden_state = torch.stack(next_hidden_layers)
            last_layer_hidden_for_head = self.dropout(posterior_hidden_state[-1])

            # Use the updated hidden state to predict the token (for reconstruction loss)
            logits = self.reconstruction_head(last_layer_hidden_for_head)
            all_reconstruction_logits.append(logits)

            # Scheduled sampling logic
            if self.training and ground_truth_tokens is not None:
                # --- Scheduled Sampling (Training Mode) ---
                # This logic must be graph-compatible for torch.compile.

                # Generate a random tensor for the decision.
                # Shape: [batch_size, 1] to allow per-item decisions.
                prob_tensor = torch.rand(batch_size, 1, device=device)
                use_teacher_force_mask = prob_tensor < teacher_forcing_prob

                # Prepare both branches of the decision.
                teacher_tokens = ground_truth_tokens[:, token_idx]
                sampled_tokens = torch.distributions.Categorical(logits=logits).sample()

                # Use torch.where to select based on the mask.
                # The mask is broadcasted to match the token tensor shapes.
                next_token_indices = torch.where(use_teacher_force_mask.squeeze(-1), teacher_tokens, sampled_tokens)
            else:
                # --- Autoregressive Sampling (Inference Mode) ---
                # Always sample from the model's own predictions.
                next_token_indices = torch.distributions.Categorical(logits=logits).sample()

            next_token_embed_raw = self.token_embedding(next_token_indices)
            # next_token_embed_projected_raw is [batch_size, hidden_dim]
            next_token_embed_projected_raw = self.token_proj(next_token_embed_raw)

            # Add positional encoding to the current token's embedding for the next step's input
            # Unsqueeze to [batch_size, 1, hidden_dim] for pos_encoder
            next_token_embed_projected_unsqueezed = next_token_embed_projected_raw.unsqueeze(1)
            next_token_embed_with_pe = self.pos_encoder(next_token_embed_projected_unsqueezed, current_pos_idx)
            # Squeeze back to [batch_size, hidden_dim] and apply dropout
            prev_token_embed_projected = self.dropout(next_token_embed_with_pe.squeeze(1))
            current_pos_idx += 1

        # --- Generate Predictions from final state ---
        # All predictions are based on the final, corrected posterior state.
        final_hidden_state = posterior_hidden_state
        last_layer_final_hidden = self.dropout(final_hidden_state[-1])

        predicted_reward = self.reward_head(last_layer_final_hidden)
        predicted_done_logits = self.done_head(last_layer_final_hidden)

        reconstructed_logits_flat = torch.stack(all_reconstruction_logits, dim=1)
        reconstructed_logits = reconstructed_logits_flat.view(
            batch_size, self.grid_size, self.grid_size, self.codebook_size
        )

        return reconstructed_logits, predicted_reward, predicted_done_logits, final_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Helper function to get a zero-initialized hidden state for all GRU layers."""
        # Shape: [num_gru_layers, batch_size, hidden_dim]
        return torch.zeros(self.num_gru_layers, batch_size, self.hidden_dim, device=device)

    def encode_observation(self, tokens: torch.Tensor, prev_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Encodes a sequence of observation tokens into a new hidden state.
        This is used to "prime" the model with a real starting state.

        Args:
            tokens (torch.Tensor): The ground truth tokens for the observation.
                                   Shape: [batch_size, num_tokens]
            prev_hidden_state (torch.Tensor): The hidden state from the previous step.
                                              Shape: [num_gru_layers, batch_size, hidden_dim]

        Returns:
            torch.Tensor: The updated hidden state after processing the tokens.
                          Shape: [num_gru_layers, batch_size, hidden_dim]
        """
        batch_size = tokens.size(0)
        # Start with a learnable token, similar to the generation process
        current_pos_idx = 0
        expanded_start_token = self.start_token_embed.expand(batch_size, -1, -1)
        start_token_with_pe = self.pos_encoder(expanded_start_token, current_pos_idx)
        prev_token_embed_projected = self.dropout(start_token_with_pe.squeeze(1))
        current_pos_idx += 1

        generation_hidden_state_stack = prev_hidden_state

        for token_idx in range(self.num_tokens):
            current_gru_input_for_stack = prev_token_embed_projected
            next_hidden_layers = []

            for l_idx in range(self.num_gru_layers):
                h_prev_layer = generation_hidden_state_stack[l_idx]
                h_next_layer = self.grus[l_idx](current_gru_input_for_stack, h_prev_layer)

                if l_idx < self.num_gru_layers - 1:
                    current_gru_input_for_stack = self.dropout(h_next_layer)
                else:
                    current_gru_input_for_stack = h_next_layer
                next_hidden_layers.append(h_next_layer)

            generation_hidden_state_stack = torch.stack(next_hidden_layers)

            # Get the next token from the provided ground-truth sequence
            next_token_indices = tokens[:, token_idx]
            next_token_embed_raw = self.token_embedding(next_token_indices)
            next_token_embed_projected_raw = self.token_proj(next_token_embed_raw)

            # Add positional encoding for the next step
            next_token_embed_projected_unsqueezed = next_token_embed_projected_raw.unsqueeze(1)
            next_token_embed_with_pe = self.pos_encoder(next_token_embed_projected_unsqueezed, current_pos_idx)
            prev_token_embed_projected = self.dropout(next_token_embed_with_pe.squeeze(1))
            current_pos_idx += 1

        return generation_hidden_state_stack


# --- Usage Example ---
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    BATCH_SIZE = 32
    LATENT_DIM = 64  # VQVAE embedding dim
    ACTION_DIM_EXAMPLE = 3
    GRU_HIDDEN_DIM_EXAMPLE = 256  # Per layer
    NUM_GRU_LAYERS_EXAMPLE = 2
    DROPOUT_RATE_EXAMPLE = 0.1  # Example dropout rate
    CODEBOOK_SIZE_EXAMPLE = 512
    GRID_SIZE_EXAMPLE = 4
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
        dropout_rate=DROPOUT_RATE_EXAMPLE  # New parameter
    ).to(DEVICE_EXAMPLE)

    print(f"World Model created on device: {DEVICE_EXAMPLE}")
    print(f"Number of parameters: {sum(p.numel() for p in world_model_example.parameters()):,}")

    # --- Create Dummy Input Data ---
    action_dummy = torch.randn(BATCH_SIZE, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    # Initial hidden state now has shape [num_layers, batch_size, hidden_dim]
    hidden_state_dummy = world_model_example.get_initial_hidden_state(BATCH_SIZE, DEVICE_EXAMPLE)
    print(f"Initial hidden_state_dummy shape: {hidden_state_dummy.shape}")

    # Dummy tokens for current observation
    current_obs_tokens_dummy = torch.randint(0, CODEBOOK_SIZE_EXAMPLE,
                                             (BATCH_SIZE, GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE)).to(DEVICE_EXAMPLE)
    # For teacher forcing during training
    ground_truth_tokens_dummy = torch.randint(0, CODEBOOK_SIZE_EXAMPLE,
                                              (BATCH_SIZE, GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE)).to(DEVICE_EXAMPLE)

    # --- Perform a Forward Pass (Training with Teacher Forcing) ---
    print("\n--- Running in Training Mode (Teacher Forcing) ---")
    predicted_logits, reward, done_logits, next_hidden_stack = world_model_example(
        action_dummy,
        hidden_state_dummy,
        current_obs_tokens_dummy,
        ground_truth_tokens=ground_truth_tokens_dummy
    )
    print(f"Predicted Logits Shape: {predicted_logits.shape}")  # Should be [B, G, G, Codebook]
    print(f"Predicted Reward Shape:  {reward.shape}")  # Should be [B, 1]
    print(f"Predicted Done Shape:    {done_logits.shape}")  # Should be [B, 1]
    print(f"Next Hidden State Stack Shape: {next_hidden_stack.shape}")  # Should be [NumLayers, B, HiddenDim]

    # --- Perform a Forward Pass (Inference without Teacher Forcing) ---
    print("\n--- Running in Inference Mode ---")
    predicted_logits_inf, _, _, _ = world_model_example(
        action_dummy,
        hidden_state_dummy,
        current_obs_tokens_dummy,
        ground_truth_tokens=None
    )
    print(f"Predicted Logits Shape (Inference): {predicted_logits_inf.shape}")
