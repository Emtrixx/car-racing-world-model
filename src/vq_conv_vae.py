import torch
import torch.nn as nn
import torch.nn.functional as F

# You can adjust these based on your specific needs
IMG_CHANNELS = 3
# IMG_CHANNELS = 1  # For grayscale images
IMG_SIZE = 64
# VQ-VAE Hyperparameters
# The embedding_dim must match the output channels of the Encoder
VQVAE_EMBEDDING_DIM = 256
# The number of discrete codes in the codebook (K)
VQVAE_NUM_EMBEDDINGS = 512
# The commitment cost is a weighting factor for the commitment loss term
COMMITMENT_COST = 0.25
GRID_SIZE = 8  # new one is 8 old one is 4
# Number of hidden units in the ResNet blocks.
NUM_HIDDENS = 128
# Number of ResNet blocks.
NUM_RESIDUAL_LAYERS = 2


class VectorQuantizer(nn.Module):
    """
    Implementation of the Vector Quantizer layer.
    This layer takes the output of the encoder, finds the closest codebook embedding,
    and passes that embedding to the decoder.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # Initialize the codebook
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs shape: (B, C, H, W)
        # C should be equal to self._embedding_dim
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input to (B*H*W, C)
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances between flattened input and codebook vectors
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Find the closest codebook vector indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the flattened input
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Calculate the loss terms
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight-Through Estimator
        # This allows gradients to be copied from `quantized` to `inputs`
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)

        # Perplexity is a measure of how many codes are being used
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape `quantized` to match the decoder input format
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class ResidualBlock(nn.Module):
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
    def __init__(self, in_channels=IMG_CHANNELS, embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS,
                 num_hiddens=NUM_HIDDENS, num_residual_layers=NUM_RESIDUAL_LAYERS, commitment_cost=COMMITMENT_COST):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, embedding_dim)
        self._pre_vq_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, in_channels)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encoding_indices = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return x_recon, loss, quantized, encoding_indices, z, perplexity
