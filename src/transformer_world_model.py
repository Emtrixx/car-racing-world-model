import math
import torch
from torch import nn

from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, GRID_SIZE

# --- Default Hyperparameters ---
TRANSFORMER_EMBED_DIM = 512
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_NUM_LAYERS = 10
TRANSFORMER_FF_DIM = 2048  # Typically 4 * embed_dim
TRANSFORMER_DROPOUT_RATE = 0.1
TRANSFORMER_MAX_SEQ_LEN = 4096  # Maximum sequence length for positional encoding


# --- Positional Encoding ---
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
        seq_len = x.size(1)
        if offset + seq_len > self.pe.size(1):
            raise IndexError(f"Offset {offset} + seq_len {seq_len} is out of range for "
                             f"max_len {self.pe.size(1)} in PositionalEncoding.")
        return x + self.pe[:, offset: offset + seq_len, :]


class CustomTransformerDecoder(nn.Module):
    """
    Custom Transformer Decoder that can return attention weights.
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            type(decoder_layer)(
                d_model=decoder_layer.self_attn.embed_dim,
                nhead=decoder_layer.self_attn.num_heads,
                dim_feedforward=decoder_layer.linear1.out_features,
                dropout=decoder_layer.dropout.p,
                activation=decoder_layer.activation,
                batch_first=True,
                norm_first=False
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, need_weights=False, average_attn_weights=False):
        """
        Forward pass for the decoder.

        This implementation manually performs the forward pass of each decoder layer
        when `need_weights` is True. This is necessary because the standard
        `nn.TransformerDecoderLayer` does not return attention weights from its
        `forward` method. By replicating the layer's logic, we can capture both
        self-attention and cross-attention weights for visualization and analysis.
        """
        output = tgt
        self_attns = []
        cross_attns = []

        for mod in self.layers:
            if need_weights:
                # Manually perform forward of layer to get weights
                sa_output, sa_weights = mod.self_attn(
                    output, output, output,
                    attn_mask=tgt_mask,
                    need_weights=True, average_attn_weights=average_attn_weights
                )
                output = mod.norm1(output + mod.dropout1(sa_output))

                mha_output, mha_weights = mod.multihead_attn(
                    output, memory, memory,
                    attn_mask=memory_mask,
                    need_weights=True, average_attn_weights=average_attn_weights
                )
                output = mod.norm2(output + mod.dropout2(mha_output))
                output = mod.norm3(output + mod._ff_block(output))

                self_attns.append(sa_weights)
                cross_attns.append(mha_weights)
            else:
                output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            return output, {'self': self_attns, 'cross': cross_attns}
        else:
            return output, None


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
            max_seq_len: int = TRANSFORMER_MAX_SEQ_LEN,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_tokens_per_state = grid_size * grid_size
        self.num_queries_per_step = self.num_tokens_per_state + 1  # +1 for global token
        self.codebook_size = codebook_size
        self.action_dim = action_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.vqvae_embed_dim = vqvae_embed_dim
        self.attention_maps = None  # For storing attention maps if needed

        # --- Embedding Layers ---
        self.token_embedding = nn.Embedding(codebook_size, vqvae_embed_dim)
        self.input_projection = nn.Linear(vqvae_embed_dim, embed_dim) if vqvae_embed_dim != embed_dim else nn.Identity()
        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        # --- Learnable Embeddings ---
        self.grid_pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens_per_state, embed_dim))
        self.output_token_queries = nn.Parameter(torch.randn(1, self.num_queries_per_step, embed_dim))

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout_rate, batch_first=True, norm_first=False
        )
        # decoder_norm = nn.LayerNorm(embed_dim) if num_layers > 0 else None
        decoder_norm = None  # todo: maybe retrain with layernorm
        self.transformer_decoder = CustomTransformerDecoder(decoder_layer, num_layers, norm=decoder_norm)

        # --- Output Prediction Heads ---
        self.next_latent_head = nn.Linear(embed_dim, codebook_size)
        self.reward_head = nn.Linear(embed_dim, 1)
        self.done_head = nn.Linear(embed_dim, 1)

        # --- Mask Cache ---
        self._mask_cache = {}

    def _generate_tbtf_masks(self, history_len: int, device: torch.device):
        """
        Generates the self-attention and cross-attention masks for Temporal Block Teacher Forcing (T-BTF).
        Caches the masks for performance.
        """
        if history_len in self._mask_cache:
            return self._mask_cache[history_len]

        total_query_len = history_len * self.num_queries_per_step
        total_memory_len = history_len * (self.num_tokens_per_state + 1)

        # Self-Attention Mask (tgt_mask)
        # Prevents queries for step t from attending to queries for step t+1.
        # Allows full attention within a block of queries for a single step.
        tgt_mask = torch.ones(total_query_len, total_query_len, device=device)
        for i in range(history_len):
            for j in range(history_len):
                if j > i:
                    tgt_mask[
                        i * self.num_queries_per_step:(i + 1) * self.num_queries_per_step,
                        j * self.num_queries_per_step:(j + 1) * self.num_queries_per_step
                    ] = 0
        tgt_mask = torch.log(tgt_mask)  # Log-transform for PyTorch

        # Cross-Attention Mask (memory_mask)
        # Prevents queries for step t from attending to memory from step t+1 onwards.
        mem_mask = torch.ones(total_query_len, total_memory_len, device=device)
        for i in range(history_len):  # Target time step
            # Queries for step i can see memory up to and including step i.
            # So, we mask out memory from step i+1 onwards.
            start_col_to_mask = (i + 1) * (self.num_tokens_per_state + 1)
            mem_mask[
                i * self.num_queries_per_step:(i + 1) * self.num_queries_per_step,
                start_col_to_mask:
            ] = 0
        memory_mask = torch.log(mem_mask)

        # Cache the masks
        self._mask_cache[history_len] = (tgt_mask, memory_mask)
        return tgt_mask, memory_mask

    def forward(self, action_history: torch.Tensor, latent_token_history: torch.Tensor):
        """
        Performs Temporal Block Teacher Forcing (T-BTF).
        For a history of length H, predicts the next state, reward, and done for every step from 1 to H.

        Args:
            action_history (torch.Tensor): Shape [B, H, action_dim]
            latent_token_history (torch.Tensor): Shape [B, H, num_tokens_per_state]
        """
        batch_size, history_len = action_history.shape[0], action_history.shape[1]
        device = action_history.device

        # --- Prepare Memory Sequence (History) ---
        state_hist_emb = self.input_projection(self.token_embedding(latent_token_history))
        state_hist_emb = state_hist_emb + self.grid_pos_embedding.unsqueeze(1)
        action_hist_emb = self.action_embedding(action_history).unsqueeze(2)
        memory_interleaved = torch.cat([state_hist_emb, action_hist_emb], dim=2)

        # Reshape memory to a single long sequence: [B, H * (num_tokens+1), D]
        memory_seq = memory_interleaved.view(batch_size, history_len * (self.num_tokens_per_state + 1), self.embed_dim)
        memory_seq = self.pos_encoder(memory_seq)
        memory_seq = self.dropout(memory_seq)

        # --- Prepare Target Sequence (Parallel Queries) ---
        # Repeat the learnable queries for each step in the history
        tgt_seq = self.output_token_queries.repeat(1, history_len, 1)  # [1, H * num_queries, D]

        # Add spatial pos encoding to the token queries part of each block
        tgt_reshaped = tgt_seq.view(1, history_len, self.num_queries_per_step, self.embed_dim)
        tgt_tokens = tgt_reshaped[:, :, :-1, :] + self.grid_pos_embedding.unsqueeze(1)
        tgt_global = tgt_reshaped[:, :, -1, :].unsqueeze(2)
        tgt_with_spatial_pos = torch.cat([tgt_tokens, tgt_global], dim=2)
        tgt_seq = tgt_with_spatial_pos.view(1, history_len * self.num_queries_per_step, self.embed_dim)

        # Add temporal pos encoding and expand for batch
        tgt_seq = self.pos_encoder(tgt_seq)
        tgt_seq = tgt_seq.expand(batch_size, -1, -1)  # [B, H * num_queries, D]
        tgt_seq = self.dropout(tgt_seq)

        # --- Generate Masks ---
        tgt_mask, memory_mask = self._generate_tbtf_masks(history_len, device)

        # --- Run Decoder ---
        decoder_output, _ = self.transformer_decoder(
            tgt=tgt_seq,
            memory=memory_seq,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )  # Shape: [B, H * num_queries, D]

        # --- Extract Predictions ---
        # Reshape output to separate the time dimension from the query dimension
        output_reshaped = decoder_output.view(batch_size, history_len, self.num_queries_per_step, self.embed_dim)

        # Predictions for each step in the history
        global_token_output = output_reshaped[:, :, -1, :]  # [B, H, D]
        predicted_rewards = self.reward_head(global_token_output)  # [B, H, 1]
        predicted_dones = self.done_head(global_token_output)  # [B, H, 1]

        token_queries_output = output_reshaped[:, :, :-1, :]  # [B, H, num_tokens, D]
        predicted_latent_logits = self.next_latent_head(token_queries_output)  # [B, H, num_tokens, codebook_size]

        # Reshape logits to match the grid structure for loss calculation
        predicted_latent_logits_grid = predicted_latent_logits.view(
            batch_size, history_len, self.grid_size, self.grid_size, self.codebook_size
        )

        # For inference, we only care about the prediction from the final step
        final_step_logits = predicted_latent_logits[:, -1, :, :]  # [B, num_tokens, codebook_size]
        generated_tokens_indices = torch.argmax(final_step_logits, dim=-1)  # [B, num_tokens]

        return predicted_latent_logits_grid, predicted_rewards, predicted_dones, generated_tokens_indices

    @torch.no_grad()
    def generate(
            self,
            action_history: torch.Tensor,  # Shape: [B, H, action_dim]
            latent_token_history: torch.Tensor,  # Shape: [B, H, num_tokens_per_state]
            get_attention: bool = False
    ):
        """
        Efficiently generates the prediction for the single next step.
        This is used for inference/dreaming, where T-BTF is not needed.
        Args:
            action_history: The history of actions.
            latent_token_history: The history of VQ-VAE token grids.
            get_attention: If True, returns attention maps from the decoder.
        Returns:
            A tuple of (next_token_indices, reward, done).
        """
        batch_size, history_len = action_history.shape[0], action_history.shape[1]

        # --- Prepare Memory Sequence (same as in forward) ---
        state_hist_emb = self.input_projection(self.token_embedding(latent_token_history))
        state_hist_emb = state_hist_emb + self.grid_pos_embedding.unsqueeze(1)
        action_hist_emb = self.action_embedding(action_history).unsqueeze(2)
        memory_interleaved = torch.cat([state_hist_emb, action_hist_emb], dim=2)
        memory_seq = memory_interleaved.view(batch_size, history_len * (self.num_tokens_per_state + 1), self.embed_dim)
        memory_seq = self.pos_encoder(memory_seq)
        # Note: No dropout during inference

        # --- Prepare Target Sequence (single query block) ---
        # Unlike in forward(), we only need one set of queries for the single next step.
        tgt_seq = self.output_token_queries.expand(batch_size, -1, -1)
        decoder_input_tokens = tgt_seq[:, :-1, :] + self.grid_pos_embedding
        decoder_input_global = tgt_seq[:, -1, :].unsqueeze(1)
        tgt_seq = torch.cat([decoder_input_tokens, decoder_input_global], dim=1)
        tgt_seq = self.pos_encoder(tgt_seq, offset=0)

        # --- Run Decoder (no masks needed for single-step prediction) ---
        decoder_output, attention_maps = self.transformer_decoder(
            tgt=tgt_seq,
            memory=memory_seq,
            tgt_mask=None,
            memory_mask=None,
            need_weights=get_attention,
            average_attn_weights=False
        )
        if get_attention:
            self.attention_maps = attention_maps

        # --- Extract Predictions ---
        global_token_output = decoder_output[:, -1, :]  # [B, D]
        predicted_reward = self.reward_head(global_token_output)  # [B, 1]
        predicted_done_logits = self.done_head(global_token_output)  # [B, 1]

        token_queries_output = decoder_output[:, :-1, :]  # [B, num_tokens, D]
        predicted_latent_logits = self.next_latent_head(token_queries_output)  # [B, num_tokens, codebook_size]

        # --- Sample next tokens ---
        # We sample from the distribution, not just argmax, for stochasticity in dreaming.
        probs = torch.softmax(predicted_latent_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs_normalized = probs / torch.clamp(probs_sum, min=1e-12)
        uniform_probs = torch.full_like(probs_normalized, 1.0 / self.codebook_size)
        invalid_mask = probs_sum <= 0
        safe_probs = torch.where(invalid_mask, uniform_probs, probs_normalized)
        # Reshape for multinomial sampling: [B * num_tokens, codebook_size]
        probs_flat = safe_probs.view(-1, self.codebook_size)
        next_tokens_flat = torch.multinomial(probs_flat, 1)
        # Reshape back to [B, num_tokens]
        generated_tokens_indices = next_tokens_flat.view(batch_size, self.num_tokens_per_state)

        # Sigmoid on done logits to get probability
        predicted_done = torch.sigmoid(predicted_done_logits) > 0.5

        return generated_tokens_indices, predicted_reward, predicted_done


# --- Usage Example ---
if __name__ == '__main__':
    # --- Configuration ---
    BATCH_SIZE = 4
    HISTORY_LEN = 8
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

    print(f"--- Running WorldModelTransformer T-BTF Example ---")
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
    world_model_tf.train()  # Set to train mode to check shapes
    num_params = sum(p.numel() for p in world_model_tf.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")

    # --- Test: Training Forward Pass ---
    print("\n--- Test: Training Forward Pass ---")
    action_hist = torch.randn(BATCH_SIZE, HISTORY_LEN, ACTION_DIM_EXAMPLE).to(DEVICE_EXAMPLE)
    tokens_hist = torch.randint(
        0, CODEBOOK_SIZE_EXAMPLE,
        (BATCH_SIZE, HISTORY_LEN, NUM_TOKENS_EXAMPLE)
    ).to(DEVICE_EXAMPLE)

    print(f"Input action history shape: {action_hist.shape}")
    print(f"Input token history shape: {tokens_hist.shape}")

    # The model now returns predictions for every step in the history
    logits_grid, rewards, dones, _ = world_model_tf(action_hist, tokens_hist)

    print("\n--- Output Shapes (Training) ---")
    print(f"Predicted logits grid shape: {logits_grid.shape}")
    print(f"Predicted rewards shape: {rewards.shape}")
    print(f"Predicted dones shape: {dones.shape}")

    # Verification
    expected_logits_shape = (BATCH_SIZE, HISTORY_LEN, GRID_SIZE_EXAMPLE, GRID_SIZE_EXAMPLE, CODEBOOK_SIZE_EXAMPLE)
    assert logits_grid.shape == expected_logits_shape, "Logits shape mismatch!"
    print("Logits shape: CORRECT")

    expected_rewards_shape = (BATCH_SIZE, HISTORY_LEN, 1)
    assert rewards.shape == expected_rewards_shape, "Rewards shape mismatch!"
    print("Rewards shape: CORRECT")

    expected_dones_shape = (BATCH_SIZE, HISTORY_LEN, 1)
    assert dones.shape == expected_dones_shape, "Dones shape mismatch!"
    print("Dones shape: CORRECT")
    print("--- T-BTF Training Test PASSED ---")

    # --- Test: Inference with generate() ---
    print("\n--- Test: Inference with generate() ---")
    world_model_tf.eval()
    with torch.no_grad():
        gen_tokens, gen_reward, gen_done, _ = world_model_tf.generate(action_hist, tokens_hist)
    print(f"Generated tokens shape: {gen_tokens.shape}")
    print(f"Generated reward shape: {gen_reward.shape}")
    print(f"Generated done shape: {gen_done.shape}")
    assert gen_tokens.shape == (BATCH_SIZE, NUM_TOKENS_EXAMPLE)
    print("--- Inference Test PASSED ---")

    # --- Test: Inference with Attention Maps ---
    print("\n--- Test: Inference with Attention Maps ---")
    world_model_tf.eval()
    with torch.no_grad():
        _, _, _ = world_model_tf.generate(action_hist, tokens_hist, get_attention=True)
        attention_maps = world_model_tf.attention_maps

    assert attention_maps is not None
    print(f"Attention maps keys: {attention_maps.keys()}")
    print(f"Number of self-attention maps (layers): {len(attention_maps['self'])}")
    print(f"Number of cross-attention maps (layers): {len(attention_maps['cross'])}")

    # Check shape of one attention map (per-head attention)
    # Shape: [B, num_heads, query_len, key_len]
    self_attn_shape = attention_maps['self'][0].shape
    cross_attn_shape = attention_maps['cross'][0].shape
    print(f"Self-attention map shape (layer 0): {self_attn_shape}")
    print(f"Cross-attention map shape (layer 0): {cross_attn_shape}")

    expected_self_attn_shape = (BATCH_SIZE, NUM_HEADS_EXAMPLE, world_model_tf.num_queries_per_step,
                                world_model_tf.num_queries_per_step)
    assert self_attn_shape == expected_self_attn_shape, "Self-attention shape mismatch!"
    print("Self-attention shape: CORRECT")

    expected_cross_attn_shape = (BATCH_SIZE, NUM_HEADS_EXAMPLE, world_model_tf.num_queries_per_step,
                                 HISTORY_LEN * (NUM_TOKENS_EXAMPLE + 1))
    assert cross_attn_shape == expected_cross_attn_shape, "Cross-attention shape mismatch!"
    print("Cross-attention shape: CORRECT")
    print("--- Attention Map Test PASSED ---")
