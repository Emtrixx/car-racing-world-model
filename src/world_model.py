import torch
from torch import nn as nn

from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS

GRU_HIDDEN_DIM = 512  # Default hidden dimension for GRU layers
GRID_SIZE = 4  # Default grid size for the world model
GRU_NUM_LAYERS = 3  # Default number of GRU layers
CODEBOOK_SIZE = 2048  # Default size of the VQ-VAE codebook


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

    def forward(self, action: torch.Tensor, prev_hidden_state: torch.Tensor, ground_truth_tokens: torch.Tensor = None):
        """
        Performs a single step of the world model prediction autoregressively.

        Args:
            action (torch.Tensor): The action taken. Shape: [batch_size, action_dim]
            prev_hidden_state (torch.Tensor): The previous hidden states of the GRUs.
                                              Shape: [num_gru_layers, batch_size, hidden_dim]
            ground_truth_tokens (torch.Tensor, optional): The ground truth tokens of the
                next state for teacher forcing. Shape: [batch_size, 16].
                If None, the model uses its own predictions (inference).

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
        next_hidden_layers_t = []
        for i in range(self.num_gru_layers):
            h_next_layer = self.grus[i](current_input, prev_hidden_state[i])
            if i < self.num_gru_layers - 1:  # Apply dropout between GRU layers
                current_input = self.dropout(h_next_layer)
            else:  # No dropout after the last GRU layer's output if it goes to heads
                current_input = h_next_layer
            next_hidden_layers_t.append(h_next_layer)  # Store original output for hidden state stack

        transition_hidden_state_stack = torch.stack(next_hidden_layers_t)
        last_layer_transition_hidden_raw = transition_hidden_state_stack[-1]
        # Dropout before prediction heads
        last_layer_transition_hidden = self.dropout(last_layer_transition_hidden_raw)

        # --- Predict Immediate Outcomes ---
        predicted_reward = self.reward_head(last_layer_transition_hidden)
        predicted_done_logits = self.done_head(last_layer_transition_hidden)

        # --- Autoregressive Generation Step ---
        all_logits = []
        generation_hidden_state_stack = transition_hidden_state_stack  # Start with hidden state from transition

        # Initial input for the first token generation (e.g., projected zero embedding or learned start token)
        # Applying dropout to the initial projected token (if any)
        prev_token_embed_projected = self.dropout(torch.zeros(batch_size, self.hidden_dim, device=device))

        for token_idx in range(self.num_tokens):
            current_gru_input_for_stack = prev_token_embed_projected  # Input to the first GRU layer
            next_hidden_layers_g = []

            for l_idx in range(self.num_gru_layers):
                h_prev_layer_g = generation_hidden_state_stack[l_idx]
                h_next_layer_g = self.grus[l_idx](current_gru_input_for_stack, h_prev_layer_g)

                if l_idx < self.num_gru_layers - 1:  # Apply dropout between GRU layers
                    current_gru_input_for_stack = self.dropout(h_next_layer_g)
                else:  # No dropout after the last GRU layer's output if it goes to heads
                    current_gru_input_for_stack = h_next_layer_g
                next_hidden_layers_g.append(h_next_layer_g)  # Store original output

            generation_hidden_state_stack = torch.stack(next_hidden_layers_g)
            last_layer_generation_hidden_raw = generation_hidden_state_stack[-1]
            # Dropout before the latent prediction head
            last_layer_generation_hidden = self.dropout(last_layer_generation_hidden_raw)

            current_logits = self.next_latent_head(last_layer_generation_hidden)
            all_logits.append(current_logits)

            if ground_truth_tokens is not None:  # Teacher forcing
                next_token_indices = ground_truth_tokens[:, token_idx]
            else:  # Inference
                next_token_indices = torch.distributions.Categorical(logits=current_logits).sample()

            next_token_embed_raw = self.token_embedding(next_token_indices)
            next_token_embed_projected_raw = self.token_proj(next_token_embed_raw)
            # Apply dropout to the projected embedding of the chosen/true token for the next step
            prev_token_embed_projected = self.dropout(next_token_embed_projected_raw)

        predicted_latent_logits_flat = torch.stack(all_logits, dim=1)
        predicted_latent_logits = predicted_latent_logits_flat.view(
            batch_size, self.grid_size, self.grid_size, self.codebook_size
        )

        final_hidden_state_stack = generation_hidden_state_stack

        return predicted_latent_logits, predicted_reward, predicted_done_logits, final_hidden_state_stack

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Helper function to get a zero-initialized hidden state for all GRU layers."""
        # Shape: [num_gru_layers, batch_size, hidden_dim]
        return torch.zeros(self.num_gru_layers, batch_size, self.hidden_dim, device=device)


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

    # For teacher forcing during training
    ground_truth_tokens_dummy = torch.randint(0, CODEBOOK_SIZE_EXAMPLE,
                                              (BATCH_SIZE, GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE)).to(DEVICE_EXAMPLE)

    # --- Perform a Forward Pass (Training with Teacher Forcing) ---
    print("\n--- Running in Training Mode (Teacher Forcing) ---")
    predicted_logits, reward, done_logits, next_hidden_stack = world_model_example(
        action_dummy, hidden_state_dummy, ground_truth_tokens=ground_truth_tokens_dummy
    )
    print(f"Predicted Logits Shape: {predicted_logits.shape}")  # Should be [B, G, G, Codebook]
    print(f"Predicted Reward Shape:  {reward.shape}")  # Should be [B, 1]
    print(f"Predicted Done Shape:    {done_logits.shape}")  # Should be [B, 1]
    print(f"Next Hidden State Stack Shape: {next_hidden_stack.shape}")  # Should be [NumLayers, B, HiddenDim]

    # --- Perform a Forward Pass (Inference without Teacher Forcing) ---
    print("\n--- Running in Inference Mode ---")
    predicted_logits_inf, _, _, _ = world_model_example(
        action_dummy, hidden_state_dummy, ground_truth_tokens=None
    )
    print(f"Predicted Logits Shape (Inference): {predicted_logits_inf.shape}")
