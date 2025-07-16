from typing import Optional

import torch
from torch import nn
import math
import random

from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS  # Assuming this is available

# --- Default Hyperparameters ---
TRANSFORMER_EMBED_DIM = 512
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_FF_DIM = 2048  # Typically 4 * embed_dim
TRANSFORMER_DROPOUT_RATE = 0.1
GRID_SIZE = 4  # Default grid size for the world model


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
            max_seq_len: int = 100,  # Max sequence length for pos encoding (num_tokens + 1 typically)
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
        # Positional encoding for the sequence of generated tokens + 1 for action/context
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        # --- Learnable Transition Query Token ---
        # This token is combined with the action to kickstart the process
        self.transition_query_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --- Transformer Decoder ---
        # The decoder will be used autoregressively to generate token sequences.
        # The 'memory' for the decoder will be derived from the action.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True  # Important: [batch, seq, feature]
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # --- Output Prediction Heads ---
        # Operate on the output of the Transformer decoder
        self.next_latent_head = nn.Linear(embed_dim, codebook_size)
        self.reward_head = nn.Linear(embed_dim, 1)
        self.done_head = nn.Linear(embed_dim, 1)

        # Causal mask for the decoder
        # Max length for target sequence is num_tokens
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.num_tokens)

        # Store Block TF parameters
        self.block_tf_ratio = block_tf_ratio
        self.block_size = block_size

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal mask for attending to previous positions."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(
            self,
            action: torch.Tensor,
            prev_latent_tokens: torch.Tensor,  # Expects [B, num_tokens]
    ):
        batch_size = action.size(0)

        # --- Create Memory from Previous State and Action ---
        prev_state_emb = self.token_embedding(prev_latent_tokens)  # [B, num_tokens, embed_dim]
        action_emb = self.action_embedding(action).unsqueeze(1)  # [B, 1, embed_dim]

        # Combine state and action to form the memory context
        memory = torch.cat([prev_state_emb, action_emb], dim=1)  # [B, num_tokens + 1, embed_dim]
        memory = self.pos_encoder(memory)  # Apply positional encoding to the memory sequence
        memory = self.dropout(memory)

        # --- Predict Reward and Done from this context ---
        # We can predict R/D from the action's representation within the memory
        action_context_vector = memory[:, -1, :]  # Get the embedding for the action token
        predicted_reward = self.reward_head(action_context_vector)
        predicted_done = self.done_head(action_context_vector)

        # --- Predict Next State Tokens in Parallel (Paper's BTF) ---
        # The target for the decoder is a sequence of learnable queries
        decoder_input = self.output_token_queries.expand(batch_size, -1, -1)
        # Positional encoding for the target queries is important for order
        decoder_input = self.pos_encoder(decoder_input, offset=0)

        # Single pass through the decoder to predict all tokens in parallel
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=None,  # No causality needed among output tokens
        )  # Output: [B, num_tokens, embed_dim]

        # Get logits for all tokens at once
        predicted_latent_logits_flat = self.next_latent_head(decoder_output)  # [B, num_tokens, codebook_size]

        # Reshape for downstream use
        predicted_latent_logits = predicted_latent_logits_flat.view(
            batch_size, self.grid_size, self.grid_size, self.codebook_size
        )

        # For inference/testing: get the predicted token indices
        # During training, the loss is calculated on the logits directly
        generated_tokens_indices = torch.argmax(predicted_latent_logits_flat, dim=-1)

        return predicted_latent_logits, predicted_reward, predicted_done, None, generated_tokens_indices
