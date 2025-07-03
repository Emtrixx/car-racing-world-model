import torch
from torch import nn as nn


# --- GRU-based World Model ---
class WorldModelGRU(nn.Module):
    """
    A GRU-based world model that predicts the next state, reward, and done flag.

    This model assumes the state representation is a grid of latent
    vectors from a model a VQ-VAE. It flattens this grid before
    processing it with the GRU.
    """

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            hidden_dim: int,
            codebook_size: int,
            grid_size: int = 4
    ):
        """
        Initializes the World Model layers.

        Args:
            latent_dim (int): The dimension of each latent vector in the grid.
            action_dim (int): The dimension of the action vector.
            hidden_dim (int): The dimension of the GRU's hidden state.
            codebook_size (int): The size of the VQ-VAE codebook for predicting logits.
            grid_size (int): The size of the input grid (e.g., 4 for a 4x4 map).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.codebook_size = codebook_size

        # Total dimension of the flattened 4x4 latent grid
        flattened_latent_dim = grid_size * grid_size * latent_dim

        # --- Input Processing Layers ---
        # A simple MLP to combine flattened latents and actions
        self.input_processor = nn.Sequential(
            nn.Linear(flattened_latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        # --- Recurrent Core ---
        # GRUCell processes one timestep at a time
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # --- Output Prediction Heads ---
        # These project the GRU's hidden state to the desired outputs.
        # Predicts the logits for the next latent state (4x4 grid)
        self.next_latent_head = nn.Linear(hidden_dim, grid_size * grid_size * codebook_size)

        # Predicts the scalar reward
        self.reward_head = nn.Linear(hidden_dim, 1)

        # Predicts the 'done' logit (before applying sigmoid)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, current_latents: torch.Tensor, action: torch.Tensor, hidden_state: torch.Tensor):
        """
        Performs a single step of the world model prediction.

        Args:
            current_latents (torch.Tensor): The current state's latent grid.
                                            Shape: [batch_size, grid_size, grid_size, latent_dim]
            action (torch.Tensor): The action taken in the current state.
                                   Shape: [batch_size, action_dim]
            hidden_state (torch.Tensor): The previous hidden state of the GRU.
                                         Shape: [batch_size, hidden_dim]

        Returns:
            Tuple containing:
            - predicted_latent_logits (torch.Tensor): Logits for the next latent grid.
              Shape: [batch_size, grid_size, grid_size, num_vq_codes]
            - predicted_reward (torch.Tensor): The predicted scalar reward.
              Shape: [batch_size, 1]
            - predicted_done_logits (torch.Tensor): The predicted done logit.
              Shape: [batch_size, 1]
            - next_hidden_state (torch.Tensor): The updated GRU hidden state.
              Shape: [batch_size, hidden_dim]
        """
        batch_size = current_latents.size(0)

        # Flatten the 4x4 latent grid into a single vector
        # Shape: [batch_size, 4, 4, latent_dim] -> [batch_size, 16 * latent_dim]
        flattened_latents = current_latents.view(batch_size, -1)

        # Combine flattened latents and action into a single input vector
        # Shape: [batch_size, 16 * latent_dim + action_dim]
        model_input = torch.cat([flattened_latents, action], dim=-1)

        # Process the combined input to match the GRU's input dimension
        # Shape: [batch_size, hidden_dim]
        gru_input = self.input_processor(model_input)

        # Perform one step of the GRU
        # Input: [b, hidden_dim], Hidden: [b, hidden_dim] -> Output: [b, hidden_dim]
        next_hidden_state = self.gru(gru_input, hidden_state)

        # Predict the outputs from the new hidden state
        latent_logits_flat = self.next_latent_head(next_hidden_state)
        predicted_reward = self.reward_head(next_hidden_state)
        predicted_done_logits = self.done_head(next_hidden_state)

        # Reshape the latent logits back to a 4x4 grid
        # Shape: [b, 16 * num_codes] -> [b, 4, 4, num_codes]
        predicted_latent_logits = latent_logits_flat.view(
            batch_size, self.grid_size, self.grid_size, self.codebook_size
        )

        return predicted_latent_logits, predicted_reward, predicted_done_logits, next_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Helper function to get a zero-initialized hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# --- Usage Example ---
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    BATCH_SIZE = 32
    LATENT_DIM = 64  # Dimension of each vector in the VQ-VAE grid
    ACTION_DIM = 10  # e.g., 10 discrete actions
    HIDDEN_DIM = 256  # Size of the GRU's memory
    CODEBOOK_SIZE = 512  # Size of the VQ-VAE codebook
    GRID_SIZE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Instantiate the Model ---
    world_model = WorldModelGRU(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        codebook_size=CODEBOOK_SIZE,
        grid_size=GRID_SIZE
    ).to(DEVICE)

    print(f"World Model created on device: {DEVICE}")
    print(f"Number of parameters: {sum(p.numel() for p in world_model.parameters()):,}")

    # --- Create Dummy Input Data ---
    # This simulates the output of a VQ-VAE encoder
    current_latents_dummy = torch.randn(BATCH_SIZE, GRID_SIZE, GRID_SIZE, LATENT_DIM).to(DEVICE)
    # A random action
    action_dummy = torch.randn(BATCH_SIZE, ACTION_DIM).to(DEVICE)
    # Initial hidden state for the GRU (usually all zeros at the start of an episode)
    hidden_state_dummy = world_model.get_initial_hidden_state(BATCH_SIZE, DEVICE)

    # --- Perform a Forward Pass ---
    predicted_logits, reward, done_logits, next_hidden = world_model(
        current_latents_dummy, action_dummy, hidden_state_dummy
    )

    # --- Print Output Shapes to Verify ---
    print("\n--- Input Shapes ---")
    print(f"Current Latents: {current_latents_dummy.shape}")
    print(f"Action:          {action_dummy.shape}")
    print(f"Hidden State:    {hidden_state_dummy.shape}")

    print("\n--- Output Shapes ---")
    print(f"Predicted Logits:  {predicted_logits.shape}")
    print(f"Predicted Reward:  {reward.shape}")
    print(f"Predicted Done:    {done_logits.shape}")
    print(f"Next Hidden State: {next_hidden.shape}")

    # --- How to use the outputs ---
    # Get the predicted next state by sampling from the logits
    # The temperature can be used to control the randomness of the prediction.
    predicted_latent_indices = torch.distributions.Categorical(logits=predicted_logits).sample()
    print(f"\nSampled next state indices shape: {predicted_latent_indices.shape}")

    # Get the probability of the episode being done
    done_probability = torch.sigmoid(done_logits)
    print(f"Done probability shape: {done_probability.shape}")
