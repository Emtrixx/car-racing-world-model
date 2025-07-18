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
            ff_dim: int = TRANSFORMER_FF_DIM,  # Typically 4 * embed_dim
            grid_size: int = GRID_SIZE,
            dropout_rate: float = TRANSFORMER_DROPOUT_RATE,
            max_seq_len: int = 128,  # Max sequence length for pos encoding (num_tokens + 1 typically)
            block_tf_ratio: float = 0.5,  # Added for storing from config
            block_size: int = 4  # Added for storing from config
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size
        self.codebook_size = codebook_size
        self.action_dim = action_dim

        self.dropout = nn.Dropout(dropout_rate)
        self.vqvae_embed_dim = vqvae_embed_dim  # Store for reference if needed

        # --- Embedding Layers ---
        # This embedding layer will receive the copied weights from VQ-VAE
        self.token_embedding = nn.Embedding(codebook_size, vqvae_embed_dim)

        # Projection layer if VQ-VAE's dim is different from Transformer's internal dim
        if vqvae_embed_dim != embed_dim:
            self.input_projection = nn.Linear(vqvae_embed_dim, embed_dim)
        else:
            self.input_projection = nn.Identity()

        self.action_embedding = nn.Linear(action_dim, embed_dim)
        # Positional encoding now needs to handle the full temporal sequence length
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        # --- Transformer Encoder ---
        # We now use an Encoder, as it's better suited for creating a rich context representation
        # from a sequence of inputs, using self-attention with a causal mask.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output Prediction Heads ---
        # These operate on the output of the Transformer encoder
        self.next_latent_head = nn.Linear(embed_dim, self.num_tokens * codebook_size)
        self.reward_head = nn.Linear(embed_dim, 1)
        self.done_head = nn.Linear(embed_dim, 1)

        # Store Block TF parameters
        self.block_tf_ratio = block_tf_ratio
        self.block_size = block_size

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal mask for attending to previous positions."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
            self,
            action: torch.Tensor,
            prev_latent_tokens: torch.Tensor,
    ):
        # The input is assumed to be a sequence: [B, S, ...]
        # B = batch_size, S = sequence_length
        batch_size, seq_len = action.shape[0], action.shape[1]

        # [B, S, H, W] -> [B, S, H*W]
        prev_latent_tokens_flat = prev_latent_tokens.flatten(start_dim=2)

        # --- Create Input Sequence from Previous States and Actions ---
        # [B, S, H*W] -> [B, S, H*W, vqvae_embed_dim]
        prev_state_emb = self.token_embedding(prev_latent_tokens_flat)
        # [B, S, H*W, vqvae_embed_dim] -> [B, S, H*W, embed_dim]
        prev_state_emb = self.input_projection(prev_state_emb)

        # Average the token embeddings for each state to get a single vector per state
        # [B, S, H*W, embed_dim] -> [B, S, embed_dim]
        prev_state_emb_pooled = prev_state_emb.mean(dim=2)

        # [B, S, action_dim] -> [B, S, embed_dim]
        action_emb = self.action_embedding(action)

        # Combine state and action embeddings. This is our input sequence.
        # Shape: [B, S, D]
        input_sequence = prev_state_emb_pooled + action_emb
        input_sequence = self.pos_encoder(input_sequence)
        input_sequence = self.dropout(input_sequence)

        # --- Process sequence through Transformer Encoder with Causal Mask ---
        # The causal mask ensures that the prediction for position `i` can only attend to positions `< i`.
        causal_mask = self._generate_causal_mask(seq_len, device=input_sequence.device)
        # Output shape: [B, S, D]
        context_vectors = self.transformer_encoder(input_sequence, mask=causal_mask)

        # --- Predict from the context vectors ---
        # [B, S, D] -> [B, S, 1]
        predicted_reward = self.reward_head(context_vectors)
        predicted_done = self.done_head(context_vectors)

        # [B, S, D] -> [B, S, num_tokens * codebook_size]
        predicted_latent_logits_flat = self.next_latent_head(context_vectors)

        # --- Reshape token logits to match target shape ---
        # [B, S, num_tokens * codebook_size] -> [B, S, H, W, codebook_size]
        predicted_latent_logits = predicted_latent_logits_flat.view(
            batch_size, seq_len, self.grid_size, self.grid_size, self.codebook_size
        )

        # For inference: get the predicted token indices
        # [B, S, H, W, C] -> [B, S, H*W]
        generated_tokens_indices = torch.argmax(predicted_latent_logits, dim=-1).flatten(start_dim=2)

        return predicted_latent_logits, predicted_reward, predicted_done, None, generated_tokens_indices


# --- Usage Example ---
if __name__ == '__main__':
    # --- Configuration ---
    BATCH_SIZE = 4
    SEQ_LEN = 8
    LATENT_DIM_EXAMPLE = 64
    ACTION_DIM_EXAMPLE = 3
    CODEBOOK_SIZE_EXAMPLE = 512
    GRID_SIZE_EXAMPLE = 4
    NUM_TOKENS_EXAMPLE = GRID_SIZE_EXAMPLE * GRID_SIZE_EXAMPLE
    DEVICE_EXAMPLE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformer specific params
    EMBED_DIM_EXAMPLE = 128
    NUM_HEADS_EXAMPLE = 4
    NUM_LAYERS_EXAMPLE = 2
    FF_DIM_EXAMPLE = EMBED_DIM_EXAMPLE * 4
    DROPOUT_EXAMPLE = 0.1
    MAX_SEQ_LEN_EXAMPLE = 256

    print(f"--- Running WorldModelTransformer Example ---")
    print(f"Device: {DEVICE_EXAMPLE}")

    # --- Model Initialization ---
    world_model_tf = WorldModelTransformer(
        vqvae_embed_dim=LATENT_DIM_EXAMPLE,
        action_dim=ACTION_DIM_EXAMPLE,
        codebook_size=CODEBOOK_SIZE_EXAMPLE,
        embed_dim=EMBED_DIM_EXAMPLE,
        num_heads=NUM_HEADS_EXAMPLE,
        num_layers=NUM_LAYERS_EXAMPLE,
        ff_dim=FF_DIM_EXAMPLE,
        grid_size=GRID_SIZE_EXAMPLE,
        dropout_rate=DROPOUT_EXAMPLE,
        max_seq_len=MAX_SEQ_LEN_EXAMPLE
    ).to(DEVICE_EXAMPLE)
    world_model_tf.eval()
    num_params = sum(p.numel() for p in world_model_tf.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")

    # --- Test 1: Single Step Inference (Batch Mode) ---
    print("\n--- Test 1: Single Step Inference ---")
    # For single step, sequence length (S) is 1
    action_single = torch.randn(BATCH_SIZE, 1, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    tokens_single = torch.randint(
        0, CODEBOOK_SIZE_EXAMPLE,
        (BATCH_SIZE, 1, GRID_SIZE_EXAMPLE, GRID_SIZE_EXAMPLE)
    ).to(DEVICE_EXAMPLE)

    print(f"Input action shape: {action_single.shape}")
    print(f"Input tokens shape: {tokens_single.shape}")

    with torch.no_grad():
        logits, reward, done, _, gen_tokens = world_model_tf(action_single, tokens_single)

    print("\n--- Output Shapes (Single Step) ---")
    print(f"Predicted logits shape: {logits.shape}")
    print(f"Predicted reward shape: {reward.shape}")
    print(f"Predicted done shape: {done.shape}")
    print(f"Generated tokens shape: {gen_tokens.shape}")

    # Verification
    expected_logits_shape = (BATCH_SIZE, 1, GRID_SIZE_EXAMPLE, GRID_SIZE_EXAMPLE, CODEBOOK_SIZE_EXAMPLE)
    assert logits.shape == expected_logits_shape, f"Logits shape mismatch! Expected {expected_logits_shape}, Got {logits.shape}"
    print("Logits shape: CORRECT")

    expected_reward_done_shape = (BATCH_SIZE, 1, 1)
    assert reward.shape == expected_reward_done_shape, f"Reward shape mismatch! Expected {expected_reward_done_shape}, Got {reward.shape}"
    print("Reward shape: CORRECT")
    assert done.shape == expected_reward_done_shape, f"Done shape mismatch! Expected {expected_reward_done_shape}, Got {done.shape}"
    print("Done shape: CORRECT")

    expected_tokens_shape = (BATCH_SIZE, 1, NUM_TOKENS_EXAMPLE)
    assert gen_tokens.shape == expected_tokens_shape, f"Generated tokens shape mismatch! Expected {expected_tokens_shape}, Got {gen_tokens.shape}"
    print("Generated tokens shape: CORRECT")
    print("--- Single Step Test PASSED ---")

    # --- Test 2: Sequence Data (Training Mode) ---
    print("\n--- Test 2: Sequence Data for Training ---")
    action_seq = torch.randn(BATCH_SIZE, SEQ_LEN, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    tokens_seq = torch.randint(
        0, CODEBOOK_SIZE_EXAMPLE,
        (BATCH_SIZE, SEQ_LEN, GRID_SIZE_EXAMPLE, GRID_SIZE_EXAMPLE)
    ).to(DEVICE_EXAMPLE)

    print(f"Input action sequence shape: {action_seq.shape}")
    print(f"Input token sequence shape: {tokens_seq.shape}")

    with torch.no_grad():
        logits_seq, reward_seq, done_seq, _, gen_tokens_seq = world_model_tf(action_seq, tokens_seq)

    print("\n--- Output Shapes (Sequence) ---")
    print(f"Predicted logits sequence shape: {logits_seq.shape}")
    print(f"Predicted reward sequence shape: {reward_seq.shape}")
    print(f"Predicted done sequence shape: {done_seq.shape}")
    print(f"Generated tokens sequence shape: {gen_tokens_seq.shape}")

    # Verification
    expected_logits_seq_shape = (BATCH_SIZE, SEQ_LEN, GRID_SIZE_EXAMPLE, GRID_SIZE_EXAMPLE, CODEBOOK_SIZE_EXAMPLE)
    assert logits_seq.shape == expected_logits_seq_shape, f"Logits seq shape mismatch! Expected {expected_logits_seq_shape}, Got {logits_seq.shape}"
    print("Logits sequence shape: CORRECT")

    expected_reward_done_seq_shape = (BATCH_SIZE, SEQ_LEN, 1)
    assert reward_seq.shape == expected_reward_done_seq_shape, f"Reward seq shape mismatch! Expected {expected_reward_done_seq_shape}, Got {reward_seq.shape}"
    print("Reward sequence shape: CORRECT")
    assert done_seq.shape == expected_reward_done_seq_shape, f"Done seq shape mismatch! Expected {expected_reward_done_seq_shape}, Got {done_seq.shape}"
    print("Done sequence shape: CORRECT")

    expected_tokens_seq_shape = (BATCH_SIZE, SEQ_LEN, NUM_TOKENS_EXAMPLE)
    assert gen_tokens_seq.shape == expected_tokens_seq_shape, f"Generated tokens seq shape mismatch! Expected {expected_tokens_seq_shape}, Got {gen_tokens_seq.shape}"
    print("Generated tokens sequence shape: CORRECT")
    print("--- Sequence Test PASSED ---")

    print("\n--- Full Example Run Complete ---")
