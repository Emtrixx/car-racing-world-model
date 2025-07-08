import torch
from torch import nn
from torch.nn import functional as F

# You can adjust these based on your specific needs
# IMG_CHANNELS = 3
IMG_CHANNELS = 1  # For grayscale images
IMG_SIZE = 64
# VQ-VAE Hyperparameters
# The embedding_dim must match the output channels of the Encoder
EMBEDDING_DIM = 128
# The number of discrete codes in the codebook (K)
NUM_EMBEDDINGS = 512
# The commitment cost is a weighting factor for the commitment loss term
COMMITMENT_COST = 0.25


class VectorQuantizer(nn.Module):
    """
    The core Vector-Quantization layer.
    Takes a continuous tensor from the encoder and maps it to a discrete one.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # The codebook is an embedding layer
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize the weights of the codebook
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

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

        # --- Calculate the VQ-VAE Loss ---
        # The VQ-VAE loss has two parts: the codebook loss and the commitment loss.
        # 1. Codebook Loss: Encourages codebook vectors to match encoder outputs.
        codebook_loss = F.mse_loss(quantized_latents, latents.detach())
        # 2. Commitment Loss: Encourages the encoder to commit to a codebook vector.
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # --- Straight-Through Estimator ---
        # This allows gradients to flow back to the encoder during backpropagation
        # as if the quantization step was just an identity function.
        quantized_latents = latents + (quantized_latents - latents).detach()

        # Reshape back to the original [B, C, H, W] format
        return vq_loss, quantized_latents.permute(0, 3, 1, 2).contiguous(), encoding_indices.view(latents.shape[:-1])


class Encoder(nn.Module):
    """
    The CNN Encoder.
    """

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class Decoder(nn.Module):
    """
    The CNN Decoder.
    Takes the quantized feature map and reconstructs the image.
    """

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.convT1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.convT4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.convT1(x))
        x = F.relu(self.convT2(x))
        x = F.relu(self.convT3(x))
        # Use sigmoid to ensure output pixels are in the [0, 1] range
        return torch.sigmoid(self.convT4(x))


class VQVAE(nn.Module):
    """
    The full VQ-VAE model that combines the Encoder, VectorQuantizer, and Decoder.
    """

    def __init__(self, in_channels=IMG_CHANNELS, embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS,
                 commitment_cost=COMMITMENT_COST):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, embedding_dim)
        # The VQ layer sits between the encoder and the decoder
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
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
        return x_recon, vq_loss, quantized, encoding_indices


if __name__ == '__main__':
    # Instantiate the model
    model = VQVAE()
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(16, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    # Forward pass
    reconstruction, vq_loss, quantized_output, tokens = model(dummy_input)

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
