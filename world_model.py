import torch
from torch import nn as nn

GRU_HIDDEN_DIM = 256


# --- GRU-based World Model (Autoregressive Version) ---
class WorldModelGRU(nn.Module):
    """
    An autoregressive GRU-based world model.

    This model performs two main steps:
    1. A 'transition' step where the action updates the hidden state, which is then
       used to predict the immediate reward and done flag.
    2. An 'autoregressive generation' step where it predicts the 4x4 latent
       feature map one token at a time over 16 recurrent steps.
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
            latent_dim (int): The dimension of each latent vector from the VQ-VAE.
            action_dim (int): The dimension of the action vector.
            hidden_dim (int): The dimension of the GRU's hidden state.
            codebook_size (int): The size of the VQ-VAE codebook.
            grid_size (int): The size of the input grid (e.g., 4 for a 4x4 map).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.codebook_size = codebook_size

        # --- Input Processing Layers ---
        # Embed the discrete VQ-VAE tokens and continuous actions.
        self.token_embedding = nn.Embedding(codebook_size, latent_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)

        # A projection layer to match the token embedding dimension to the GRU's input dimension.
        self.token_proj = nn.Linear(latent_dim, hidden_dim)

        # --- Recurrent Core ---
        # The GRU's input dimension is consistently `hidden_dim`.
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # --- Output Prediction Heads ---
        # Predicts logits for a SINGLE next token from the codebook.
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
            prev_hidden_state (torch.Tensor): The previous hidden state of the GRU.
                                              Shape: [batch_size, hidden_dim]
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
        action_embed = self.action_embedding(action)
        transition_hidden_state = self.gru(action_embed, prev_hidden_state)

        # --- Predict Immediate Outcomes ---
        # Use the transition state to predict reward and done status.
        predicted_reward = self.reward_head(transition_hidden_state)
        predicted_done_logits = self.done_head(transition_hidden_state)

        # --- Autoregressive Generation Step ---
        all_logits = []
        generation_hidden_state = transition_hidden_state
        # Start with a zero vector as the embedding for the first "previous" token.
        prev_token_embed = torch.zeros(batch_size, self.hidden_dim, device=device)

        for i in range(self.num_tokens):
            generation_hidden_state = self.gru(prev_token_embed, generation_hidden_state)
            current_logits = self.next_latent_head(generation_hidden_state)
            all_logits.append(current_logits)

            if ground_truth_tokens is not None:
                # --- Teacher Forcing (Training) ---
                next_token_indices = ground_truth_tokens[:, i]
            else:
                # --- Inference ---
                next_token_indices = torch.distributions.Categorical(logits=current_logits).sample()

            # Project the next token's embedding to the correct dimension for the GRU input.
            next_token_embed = self.token_embedding(next_token_indices)
            prev_token_embed = self.token_proj(next_token_embed)

        # Stack all predicted logits and reshape to the grid format.
        predicted_latent_logits_flat = torch.stack(all_logits, dim=1)  # [B, 16, codebook_size]
        predicted_latent_logits = predicted_latent_logits_flat.view(
            batch_size, self.grid_size, self.grid_size, self.codebook_size
        )

        final_hidden_state = generation_hidden_state

        return predicted_latent_logits, predicted_reward, predicted_done_logits, final_hidden_state

    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Helper function to get a zero-initialized hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# --- Usage Example ---
if __name__ == '__main__':
    # --- Model Hyperparameters ---
    BATCH_SIZE = 32
    LATENT_DIM = 64
    ACTION_DIM = 10
    CODEBOOK_SIZE = 512
    GRID_SIZE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Instantiate the Model ---
    world_model = WorldModelGRU(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=GRU_HIDDEN_DIM,
        codebook_size=CODEBOOK_SIZE, grid_size=GRID_SIZE
    ).to(DEVICE)

    print(f"World Model created on device: {DEVICE}")
    print(f"Number of parameters: {sum(p.numel() for p in world_model.parameters()):,}")

    # --- Create Dummy Input Data ---
    action_dummy = torch.randn(BATCH_SIZE, ACTION_DIM).to(DEVICE)
    hidden_state_dummy = world_model.get_initial_hidden_state(BATCH_SIZE, DEVICE)
    # For teacher forcing during training
    ground_truth_tokens_dummy = torch.randint(0, CODEBOOK_SIZE, (BATCH_SIZE, GRID_SIZE * GRID_SIZE)).to(DEVICE)

    # --- Perform a Forward Pass (Training with Teacher Forcing) ---
    print("\n--- Running in Training Mode (Teacher Forcing) ---")
    predicted_logits, reward, done_logits, next_hidden = world_model(
        action_dummy, hidden_state_dummy, ground_truth_tokens=ground_truth_tokens_dummy
    )
    print(f"Predicted Logits Shape: {predicted_logits.shape}")
    print(f"Predicted Reward Shape:  {reward.shape}")
    print(f"Predicted Done Shape:    {done_logits.shape}")
    print(f"Next Hidden State Shape: {next_hidden.shape}")

    # --- Perform a Forward Pass (Inference without Teacher Forcing) ---
    print("\n--- Running in Inference Mode ---")
    predicted_logits_inf, _, _, _ = world_model(
        action_dummy, hidden_state_dummy, ground_truth_tokens=None
    )
    print(f"Predicted Logits Shape (Inference): {predicted_logits_inf.shape}")
