import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model Configuration ---
# You can adjust these based on your specific needs
IMG_CHANNELS = 3
# IMG_CHANNELS = 1  # For grayscale images
IMG_SIZE = 64
GRID_SIZE = 8  # Size of the grid for the VQ-VAE codebook
# --- VQ-VAE Hyperparameters ---
# The embedding_dim must match the output channels of the Encoder
VQVAE_EMBEDDING_DIM = 256
# The number of discrete codes in the codebook (K)
VQVAE_NUM_EMBEDDINGS = 512
# The commitment cost is a weighting factor for the commitment loss term
COMMITMENT_COST = 0.25
# Decay for the EMA update, a value close to 1 is standard.
DECAY = 0.99
# Number of hidden units in the ResNet blocks.
NUM_HIDDENS = 128
# Number of ResNet blocks.
NUM_RESIDUAL_LAYERS = 2


class VectorQuantizerEMA(nn.Module):
    """
    SOTA Vector Quantizer layer using Exponential Moving Average (EMA) updates.
    This is the modern replacement for the original VQ-VAE's codebook loss.
    The codebook is not trained by the optimizer, but updated directly.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, decay: float,
                 epsilon: float = 1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        # Initialize the codebook embeddings. These are not trained by the optimizer.
        # We use register_buffer to make them part of the model's state_dict,
        # but not parameters.
        embeddings = torch.empty(self._num_embeddings, self._embedding_dim)
        nn.init.uniform_(embeddings, -1.0 / self._num_embeddings, 1.0 / self._num_embeddings)
        self.register_buffer("_embeddings", embeddings)

        # Buffers for tracking cluster sizes and EMA for the codebook
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", embeddings.clone())

    def forward(self, inputs):
        # inputs shape: (B, C, H, W) -> permute to (B, H, W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_permuted.shape

        # Flatten input to (B*H*W, C)
        flat_input = inputs_permuted.view(-1, self._embedding_dim)

        # Calculate distances between flattened input and codebook vectors
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embeddings ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embeddings.t()))

        # Find the closest codebook vector indices
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self._num_embeddings).float()

        # Quantize the flattened input
        quantized = torch.matmul(encodings, self._embeddings).view(input_shape)

        # EMA Codebook Update
        if self.training:
            # Update the EMA cluster size
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing to handle potential division by zero
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n
            )

            # Update the EMA weights for the codebook
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            # Update the codebook with the new EMA weights
            self._embeddings = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # The VQ loss is now only the commitment cost. The codebook loss is gone.
        commitment_loss = F.mse_loss(inputs_permuted, quantized.detach())
        loss = self._commitment_cost * commitment_loss

        # Straight-Through Estimator
        # This allows gradients to be copied from `quantized` to `inputs`
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()

        # Calculate perplexity (a measure of codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape `quantized` and `encoding_indices` to match the original spatial dimensions
        quantized_out = quantized.permute(0, 3, 1, 2).contiguous()
        encoding_indices_out = encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

        return loss, quantized_out, perplexity, encoding_indices_out


class ResidualBlock(nn.Module):
    """Standard Residual Block for the Encoder and Decoder."""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class Encoder(nn.Module):
    """The Encoder network, compresses the input image."""

    def __init__(self, in_channels, num_hiddens, num_residual_layers, embedding_dim):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = nn.ModuleList(
            [ResidualBlock(num_hiddens, num_hiddens) for _ in range(num_residual_layers)]
        )
        self._conv_4 = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        for residual_block in self._residual_stack:
            x = residual_block(x)
        return self._conv_4(F.relu(x))


class Decoder(nn.Module):
    """The Decoder network, reconstructs the image from the quantized latent space."""

    def __init__(self, in_channels, num_hiddens, num_residual_layers, out_channels):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = nn.ModuleList(
            [ResidualBlock(num_hiddens, num_hiddens) for _ in range(num_residual_layers)]
        )
        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        for residual_block in self._residual_stack:
            x = residual_block(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)


class VQVAE(nn.Module):
    """The main VQ-VAE model, now using the EMA-based Vector Quantizer."""

    def __init__(self, in_channels=IMG_CHANNELS, embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS,
                 num_hiddens=NUM_HIDDENS, num_residual_layers=NUM_RESIDUAL_LAYERS, commitment_cost=COMMITMENT_COST,
                 decay=DECAY):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, embedding_dim)
        self._pre_vq_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, stride=1)

        # Use the new VectorQuantizerEMA
        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)

        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, in_channels)

    def forward(self, x):
        # 1. Encode the input
        z = self._encoder(x)
        z = self._pre_vq_conv(z)

        # 2. Quantize the latent representation
        # The VQ layer now returns the commitment loss, the quantized tensor, perplexity, and the indices
        vq_loss, quantized, perplexity, encoding_indices = self._vq_vae(z)

        # 3. Decode the quantized representation to reconstruct the image
        x_recon = self._decoder(quantized)

        # Return the values in the requested order
        return x_recon, vq_loss, quantized, encoding_indices, z, perplexity


# --- Example Usage ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    model = VQVAE().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Create a dummy input tensor
    dummy_input = torch.randn(4, IMG_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)

    # Forward pass
    model.train()  # Set to training mode to enable EMA updates
    x_recon, vq_loss, quantized, encoding_indices, z, perplexity = model(dummy_input)

    # Print shapes and values to verify
    print("\n--- Output Verification ---")
    print(f"Input shape:          {dummy_input.shape}")
    print(f"Encoder output (z) shape: {z.shape}")
    print(f"Quantized shape:      {quantized.shape}")
    print(f"Encoding indices shape: {encoding_indices.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"VQ Loss:              {vq_loss.item():.4f}")
    print(f"Perplexity:           {perplexity.item():.4f}")
