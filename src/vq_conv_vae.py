import torch
from torch import nn
from torch.nn import functional as F

# You can adjust these based on your specific needs
# IMG_CHANNELS = 3
IMG_CHANNELS = 1  # For grayscale images
IMG_SIZE = 64
# VQ-VAE Hyperparameters
# The embedding_dim must match the output channels of the Encoder
VQVAE_EMBEDDING_DIM = 256
# The number of discrete codes in the codebook (K)
VQVAE_NUM_EMBEDDINGS = 512
# The commitment cost is a weighting factor for the commitment loss term
COMMITMENT_COST = 0.25
GRID_SIZE = 8  # new one is 8 old one is 4


class VectorQuantizer(nn.Module):
    """
    The core Vector-Quantization layer.
    Takes a continuous tensor from the encoder and maps it to a discrete one.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float, ema_decay: float = 0.99,
                 ema_epsilon: float = 1e-5):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        # The codebook is an embedding layer
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize the weights of the codebook
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        # EMA buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_dw', self.embedding.weight.data.clone())

    def forward(self, latents):
        # latents shape: [Batch, Channels, Height, Width]
        # Channels must equal embedding_dim
        latents = latents.permute(0, 2, 3, 1).contiguous()  # -> [B, H, W, C]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.embedding_dim)  # -> [B*H*W, C]

        # --- Find the closest codebook vector for each input vector ---
        # Calculate L2 distance between each latent vector and each codebook vector
        distances = (torch.sum(flat_latents ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_latents, self.embedding.weight.t()))

        # Get the index of the closest embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # -> [B*H*W, 1]

        # Convert indices to one-hot vectors
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=latents.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Look up the quantized vectors from the codebook
        quantized_latents = torch.matmul(encodings, self.embedding.weight)  # -> [B*H*W, C]
        quantized_latents = quantized_latents.view(latents_shape)  # -> [B, H, W, C]

        # --- Calculate the VQ-VAE Loss (Commitment Loss only) ---
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        vq_loss = self.commitment_cost * commitment_loss

        # --- EMA Codebook Update ---
        if self.training:
            with torch.no_grad():
                # Update EMA cluster size
                # N_k = sum_i E_k(i) where E_k(i) is 1 if x_i is assigned to code k, 0 otherwise
                cluster_counts = encodings.sum(0)  # Sum over B*H*W dimension, result shape [num_embeddings]
                self.ema_cluster_size = self.ema_decay * self.ema_cluster_size + (1 - self.ema_decay) * cluster_counts

                # Update EMA sum of vectors (dw)
                # m_k = sum_i E_k(i) * x_i
                dw = flat_latents.t() @ encodings  # [embedding_dim, B*H*W] @ [B*H*W, num_embeddings] -> [embedding_dim, num_embeddings]
                self.ema_dw = self.ema_decay * self.ema_dw + (
                        1 - self.ema_decay) * dw.t()  # dw needs to be transposed to [num_embeddings, embedding_dim]

                # Update codebook embeddings
                # e_k = m_k / N_k
                n = self.ema_cluster_size.sum()
                normalized_ema_cluster_size = (
                        (self.ema_cluster_size + self.ema_epsilon)
                        / (n + self.num_embeddings * self.ema_epsilon) * n
                )  # Laplace smoothing
                self.embedding.weight.data.copy_(self.ema_dw / normalized_ema_cluster_size.unsqueeze(1))

        # --- Straight-Through Estimator ---
        # This allows gradients to flow back to the encoder during backpropagation
        # as if the quantization step was just an identity function.
        quantized_latents = latents + (quantized_latents - latents).detach()

        # Reshape back to the original [B, C, H, W] format
        return vq_loss, quantized_latents.permute(0, 3, 1, 2).contiguous(), encoding_indices.view(latents.shape[:-1])

    @torch.no_grad()
    def reset_dead_codes(self, batch_latents):
        """
        Finds and resets dead codes. A dead code is one whose EMA cluster size
        is below a threshold. It is reset to a random encoder output from the
        current batch. This method should be called periodically during training.

        Args:
            batch_latents (torch.Tensor): The continuous output of the encoder
                                          for the current batch.
        """
        # Flatten the batch of encoder outputs
        flat_latents = batch_latents.view(-1, self.embedding_dim)

        # Find dead codes (very low usage)
        # Using a threshold of 1.0, meaning a code is dead if it's used less
        # than once on average in the EMA window.
        dead_code_indices = torch.where(self.ema_cluster_size < 1.0)[0]
        num_dead = len(dead_code_indices)

        if num_dead > 0:
            print(f"Resetting {num_dead} dead codes.")

            # Choose an equal number of random latents from the batch
            random_latents_indices = torch.randperm(flat_latents.size(0))[:num_dead]
            random_latents = flat_latents[random_latents_indices].to(self.embedding.weight.device)

            # Assign the random latents to the dead codebook entries
            self.embedding.weight.data[dead_code_indices] = random_latents

            # Reset the EMA stats for the dead codes as well
            # Give it a fresh start with a count of 1.0
            self.ema_cluster_size[dead_code_indices] = 1.0
            self.ema_dw[dead_code_indices] = random_latents


class ResidualBlock(nn.Module):
    """
    A simple residual block with two convolutional layers.
    Uses GroupNorm for stabilization.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)  # 8 groups is a common choice
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Shortcut connection if input and output channels don't match
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class Encoder(nn.Module):
    """
    The Encoder network for the VQ-VAE.
    Takes an image and downsamples it to a grid of latent embeddings.
    """

    def __init__(self, in_channels, embedding_dim):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            # Input: (B, C, 64, 64)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 32, 32)
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 16, 16)
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (B, 256, 8, 8)
            nn.ReLU(inplace=True),

            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            # Final convolution to produce the embeddings.
            nn.Conv2d(256, embedding_dim, kernel_size=1, stride=1)  # -> (B, embedding_dim, 8, 8)
        )

    def forward(self, x):
        """
        Forward pass of the encoder.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Latent feature map of shape (B, D, H', W').
        """
        return self.layers(x)


class Decoder(nn.Module):
    """
    The Decoder network for the VQ-VAE.
    Takes a quantized latent grid and upsamples it back to a full image.
    Mirrors the Encoder's architecture.
    """

    def __init__(self, embedding_dim, out_channels):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, kernel_size=1, stride=1),

            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            # Input: (B, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 16, 16)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 32, 32)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # -> (B, C, 64, 64)

            # Final activation to scale pixels to [-1, 1]
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass of the decoder.
        Args:
            x (torch.Tensor): Quantized latent tensor of shape (B, D, H', W').
        Returns:
            torch.Tensor: Reconstructed image tensor of shape (B, C, H, W).
        """
        return self.layers(x)


class VQVAE(nn.Module):
    """
    The full VQ-VAE model that combines the Encoder, VectorQuantizer, and Decoder.
    """

    def __init__(self, in_channels=IMG_CHANNELS, embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS,
                 commitment_cost=COMMITMENT_COST, ema_decay=0.99, ema_epsilon=1e-5):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, embedding_dim)
        # The VQ layer sits between the encoder and the decoder
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, ema_decay, ema_epsilon)
        self.decoder = Decoder(embedding_dim, in_channels)

    def forward(self, x):
        # 1. Encode the image to a continuous feature map
        z = self.encoder(x)
        # 2. Pass through the VQ layer to get discrete codes and the VQ loss
        vq_loss, quantized, encoding_indices = self.vq_layer(z)
        # 3. Decode the quantized representation to reconstruct the image
        x_recon = self.decoder(quantized)

        # The `encoding_indices` are the discrete tokens for your Transformer
        # Shape: [Batch, Height, Width]
        # `quantized` is the continuous representation used by PPO.
        # Shape: [Batch, EmbeddingDim, Height_feat, Width_feat]
        return x_recon, vq_loss, quantized, encoding_indices, z


if __name__ == '__main__':
    # Instantiate the model
    model = VQVAE()
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(16, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    # Forward pass
    reconstruction, vq_loss, quantized_output, tokens, _ = model(dummy_input)

    # Calculate reconstruction loss (e.g., Mean Squared Error)
    recon_loss = F.mse_loss(reconstruction, dummy_input)

    # The total loss to optimize is the sum of reconstruction and VQ losses (weighted codebook and commitment losses)
    total_loss = recon_loss + vq_loss

    print(f"Input shape:          {dummy_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Quantized output shape: {quantized_output.shape}")
    print(f"Tokens shape:         {tokens.shape} (These are the discrete latent codes)")
    print(f"Reconstruction Loss:  {recon_loss.item():.4f}")
    print(f"VQ Loss:              {vq_loss.item():.4f}")
    print(f"Total Loss:           {total_loss.item():.4f}")
