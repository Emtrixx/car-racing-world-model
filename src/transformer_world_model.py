import math

import torch
from torch import nn

from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, GRID_SIZE

# --- Default Hyperparameters ---
TRANSFORMER_EMBED_DIM = 512
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_FF_DIM = 2048  # Typically 4 * embed_dim
TRANSFORMER_DROPOUT_RATE = 0.1


# --- Positional Encoding --- todo: remove duplicate
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            offset (int): The starting position index for encoding.
        Returns:
            torch.Tensor: Tensor with added positional encoding.
        """
        # self.pe is [1, max_len, d_model]
        # x is [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        if offset + seq_len > self.pe.size(1):
            raise IndexError(f"Offset {offset} + seq_len {seq_len} is out of range for "
                             f"max_len {self.pe.size(1)} in PositionalEncoding.")
        return x + self.pe[:, offset: offset + seq_len, :]


class WorldModelTransformer(nn.Module):
    def __init__(
            self,
            vqvae_embed_dim: int,
            action_dim: int,
            codebook_size: int = VQVAE_NUM_EMBEDDINGS,
            embed_dim: int = TRANSFORMER_EMBED_DIM,
            num_heads: int = TRANSFORMER_NUM_HEADS,
            num_layers: int = TRANSFORMER_NUM_LAYERS,
            ff_dim: int = TRANSFORMER_FF_DIM,
            grid_size: int = GRID_SIZE,
            dropout_rate: float = TRANSFORMER_DROPOUT_RATE,
            max_seq_len: int = 1024,  # Maximum sequence length for positional encoding
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.codebook_size = codebook_size
        self.action_dim = action_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.vqvae_embed_dim = vqvae_embed_dim

        # --- Embedding Layers ---
        # This embedding layer will receive the copied weights from VQ-VAE
        self.token_embedding = nn.Embedding(codebook_size, vqvae_embed_dim)
        # Projection layer if VQ-VAE's dim is different from Transformer's internal dim
        self.input_projection = nn.Linear(vqvae_embed_dim, embed_dim) if vqvae_embed_dim != embed_dim else nn.Identity()
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        # --- Learnable Query Tokens for BTF ---
        # These are the queries fed into the decoder to predict the next state tokens in parallel.
        self.output_token_queries = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # --- Transformer Decoder ---
        # The decoder will be used to predict the next state tokens based on the memory context.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # --- Output Prediction Heads ---
        # Operate on the output of the Transformer decoder
        self.next_latent_head = nn.Linear(embed_dim, codebook_size)
        # Heads for reward and done prediction (operates on flattened state representation)
        flattened_dim = self.num_tokens * embed_dim
        self.reward_head = nn.Linear(flattened_dim, 1)
        self.done_head = nn.Linear(flattened_dim, 1)

    def forward(
            self,
            action_history: torch.Tensor,
            latent_token_history: torch.Tensor,
    ):
        """
        Predicts the next state given a history of actions and latent tokens.

        Args:
            action_history (torch.Tensor): Shape [B, H, action_dim]
            latent_token_history (torch.Tensor): Shape [B, H, num_tokens]
        """
        batch_size, history_len = action_history.shape[0], action_history.shape[1]

        # Embed the history of states and actions
        # [B, H, num_tokens] -> [B, H, num_tokens, vqvae_embed_dim]
        state_history_emb = self.token_embedding(latent_token_history)
        # [B, H, num_tokens, vqvae_embed_dim] -> [B, H, num_tokens, embed_dim]
        state_history_emb = self.input_projection(state_history_emb)

        # [B, H, action_dim] -> [B, H, 1, embed_dim]
        action_history_emb = self.action_embedding(action_history).unsqueeze(2)

        # Interleave states and actions to form the memory context
        # This creates a sequence of [state_t, action_t, state_t+1, action_t+1, ...]
        # Shape: [B, H, num_tokens + 1, embed_dim]
        memory_interleaved = torch.cat([state_history_emb, action_history_emb], dim=2)

        # Reshape for the transformer into a single long sequence per batch item
        # [B, H, num_tokens + 1, embed_dim] -> [B, H * (num_tokens + 1), embed_dim]
        memory = memory_interleaved.view(batch_size, history_len * (self.num_tokens + 1), self.embed_dim)

        # Apply positional encoding and dropout to the entire history
        memory = self.pos_encoder(memory)
        memory = self.dropout(memory)

        # Predict Next State Tokens in Parallel (BTF mechanism)
        # The decoder input (queries) remains the same.
        # [1, num_tokens, embed_dim] -> [B, num_tokens, embed_dim]
        decoder_input = self.output_token_queries.expand(batch_size, -1, -1)
        # Note: Positional encoding for the decoder queries starts from 0, as it's a "fresh" prediction.
        decoder_input = self.pos_encoder(decoder_input, offset=0)

        # The decoder attends to the entire historical `memory` to generate the next state.
        # Output: [B, num_tokens, embed_dim]
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=None,  # No mask needed for parallel decoding queries
        )

        # --- Predict Reward and Done from the predicted next state's context ---
        # Flatten the decoder output to create a single feature vector.
        # [B, num_tokens, embed_dim] -> [B, num_tokens * embed_dim]
        next_state_context_flat = torch.flatten(decoder_output, start_dim=1)

        # Use the flattened context for reward and done prediction
        predicted_reward = self.reward_head(next_state_context_flat)
        predicted_done = self.done_head(next_state_context_flat)

        # Get logits for all next-state tokens at once
        # [B, num_tokens, codebook_size]
        predicted_latent_logits = self.next_latent_head(decoder_output)

        # Reshape logits to match the grid structure
        # [B, H, W, codebook_size]
        predicted_latent_logits_grid = predicted_latent_logits.view(
            batch_size, self.grid_size, self.grid_size, self.codebook_size
        )

        # For inference: get the predicted token indices
        # [B, num_tokens]
        generated_tokens_indices = torch.argmax(predicted_latent_logits, dim=-1)

        return predicted_latent_logits_grid, predicted_reward, predicted_done, generated_tokens_indices


# --- Usage Example ---
if __name__ == '__main__':
    # --- Configuration ---
    BATCH_SIZE = 4
    HISTORY_LEN = 8  # We are now using a history of 8 steps
    ACTION_DIM_EXAMPLE = 3
    CODEBOOK_SIZE_EXAMPLE = 512
    GRID_SIZE_EXAMPLE = 4
    NUM_TOKENS_EXAMPLE = GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE
    DEVICE_EXAMPLE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformer specific params
    VQVAE_EMBED_DIM_EXAMPLE = 64
    EMBED_DIM_EXAMPLE = 128
    NUM_HEADS_EXAMPLE = 4
    NUM_LAYERS_EXAMPLE = 2

    print(f"--- Running WorldModelTransformer Example ---")
    print(f"Device: {DEVICE_EXAMPLE}")

    # --- Model Initialization ---
    world_model_tf = WorldModelTransformer(
        vqvae_embed_dim=VQVAE_EMBED_DIM_EXAMPLE,
        action_dim=ACTION_DIM_EXAMPLE,
        codebook_size=CODEBOOK_SIZE_EXAMPLE,
        embed_dim=EMBED_DIM_EXAMPLE,
        num_heads=NUM_HEADS_EXAMPLE,
        num_layers=NUM_LAYERS_EXAMPLE,
        grid_size=GRID_SIZE_EXAMPLE,
    ).to(DEVICE_EXAMPLE)
    world_model_tf.eval()
    num_params = sum(p.numel() for p in world_model_tf.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")

    # --- Test: Inference with History ---
    print("\n--- Test: Inference with History ---")
    # Note the new HISTORY_LEN dimension
    action_hist = torch.randn(BATCH_SIZE, HISTORY_LEN, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    tokens_hist = torch.randint(
        0, CODEBOOK_SIZE_EXAMPLE,
        (BATCH_SIZE, HISTORY_LEN, NUM_TOKENS_EXAMPLE)
    ).to(DEVICE_EXAMPLE)

    print(f"Input action history shape: {action_hist.shape}")
    print(f"Input token history shape: {tokens_hist.shape}")

    with torch.no_grad():
        logits, reward, done, gen_tokens = world_model_tf(action_hist, tokens_hist)

    print("\n--- Output Shapes ---")
    print(f"Predicted logits shape: {logits.shape}")
    print(f"Predicted reward shape: {reward.shape}")
    print(f"Predicted done shape: {done.shape}")
    print(f"Generated tokens shape: {gen_tokens.shape}")

    # Verification
    expected_logits_shape = (BATCH_SIZE, GRID_SIZE_EXAMPLE, GRID_SIZE_EXAMPLE, CODEBOOK_SIZE_EXAMPLE)
    assert logits.shape == expected_logits_shape, "Logits shape mismatch!"
    print("Logits shape: CORRECT")

    expected_reward_done_shape = (BATCH_SIZE, 1)
    assert reward.shape == expected_reward_done_shape, "Reward shape mismatch!"
    print("Reward shape: CORRECT")
    assert done.shape == expected_reward_done_shape, "Done shape mismatch!"
    print("Done shape: CORRECT")

    expected_tokens_shape = (BATCH_SIZE, NUM_TOKENS_EXAMPLE)
    assert gen_tokens.shape == expected_tokens_shape, "Generated tokens shape mismatch!"
    print("Generated tokens shape: CORRECT")
    print("--- History-Aware Test PASSED ---")
