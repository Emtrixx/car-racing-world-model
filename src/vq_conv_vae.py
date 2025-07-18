import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torch.utils.data import DataLoader, TensorDataset

# --- Model Configuration ---
# You can adjust these based on your specific needs
IMG_CHANNELS = 3
# IMG_CHANNELS = 1  # For grayscale images
IMG_SIZE = 64
GRID_SIZE = 4  # The latent grid size after encoding (4x4)

# --- VQ-VAE Hyperparameters ---
# The embedding_dim must match the output channels of the Encoder
VQVAE_EMBEDDING_DIM = 256
# The number of discrete codes in the codebook (K)
VQVAE_NUM_EMBEDDINGS = 512
# The commitment cost is a weighting factor for the commitment loss term
COMMITMENT_COST = 0.05
# Decay for the EMA update, a value close to 1 is standard.
DECAY = 0.99
# Weight for the perceptual loss term in the total loss function
PERCEPTUAL_LOSS_WEIGHT = 0.1
# Number of hidden units in the ResNet blocks.
NUM_HIDDENS = 128
# Number of ResNet blocks.
NUM_RESIDUAL_LAYERS = 2


class LPIPSLoss(nn.Module):
    """
    A wrapper for the LPIPS (perceptual loss) model.
    It calculates the perceptual distance between two images.
    """

    def __init__(self, net='vgg'):
        super(LPIPSLoss, self).__init__()
        print(f"Setting up LPIPS loss with network '{net}'...")
        # VGG is often a good choice for perceptual loss
        self.loss_fn = lpips.LPIPS(net=net)
        # We don't want to train the LPIPS model
        for param in self.loss_fn.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """
        Calculates the LPIPS loss.
        Assumes input images are in the range [0, 1].
        LPIPS model expects inputs in the range [-1, 1].
        """
        # Normalize images to the [-1, 1] range
        x_norm = x * 2 - 1
        y_norm = y * 2 - 1
        return self.loss_fn(x_norm, y_norm).mean()


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
            with torch.no_grad():
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

    def initialize_codebook_with_kmeans(self, encoder_outputs):
        """
        Initializes the codebook using K-Means clustering on a sample of encoder outputs.
        """
        print("Initializing codebook with K-Means...")
        # Simple K-Means implementation
        num_samples = encoder_outputs.size(0)

        # Randomly select initial centroids from the data points
        indices = torch.randperm(num_samples)[:self._num_embeddings]
        centroids = encoder_outputs[indices]

        for i in range(15):  # K-Means iterations
            # Assign each point to the nearest centroid
            distances = torch.cdist(encoder_outputs, centroids)
            assignments = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for j in range(self._num_embeddings):
                assigned_points = encoder_outputs[assignments == j]
                if len(assigned_points) > 0:
                    new_centroids[j] = assigned_points.mean(dim=0)
                else:  # Re-initialize centroid if it has no points
                    new_centroids[j] = encoder_outputs[torch.randint(0, num_samples, (1,))]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                print(f"K-Means converged after {i + 1} iterations.")
                break
            centroids = new_centroids

        # Update the codebook with the K-Means centroids
        self._embeddings.data.copy_(centroids)
        self._ema_w.data.copy_(centroids)
        print("Codebook initialized.")


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
    """The Encoder network, compresses a 64x64 image to a 4x4 latent map."""

    def __init__(self, in_channels, num_hiddens, num_residual_layers, embedding_dim):
        super(Encoder, self).__init__()

        # Downsampling layers
        self._downsample = nn.Sequential(
            # Input: (B, C, 64, 64)
            nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.ReLU(True),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1),  # -> 4x4
        )

        # Convolution before residual blocks
        self._pre_residual_conv = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)

        # Residual stack
        self._residual_stack = nn.ModuleList(
            [ResidualBlock(num_hiddens, num_hiddens) for _ in range(num_residual_layers)]
        )

        # Final convolution to get to embedding_dim
        self._final_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs):
        x = self._downsample(inputs)
        x = self._pre_residual_conv(x)

        for residual_block in self._residual_stack:
            x = residual_block(x)

        # The final ReLU is applied before the last conv, as in the original code
        return self._final_conv(F.relu(x))


class Decoder(nn.Module):
    """The Decoder network, reconstructs a 64x64 image from a 4x4 latent map."""

    def __init__(self, in_channels, num_hiddens, num_residual_layers, out_channels):
        super(Decoder, self).__init__()

        # Convolution before residual blocks
        self._pre_residual_conv = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)

        # Residual stack
        self._residual_stack = nn.ModuleList(
            [ResidualBlock(num_hiddens, num_hiddens) for _ in range(num_residual_layers)]
        )

        # Upsampling layers
        self._upsample = nn.Sequential(
            # Input: (B, C, 4, 4)
            nn.ConvTranspose2d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens // 2, out_channels, kernel_size=4, stride=2, padding=1)  # -> 64x64
        )

    def forward(self, inputs):
        x = self._pre_residual_conv(inputs)

        for residual_block in self._residual_stack:
            x = residual_block(x)

        return self._upsample(x)


class VQVAE(nn.Module):
    """The main VQ-VAE model, now using the EMA-based Vector Quantizer."""

    def __init__(self, in_channels=IMG_CHANNELS, embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS,
                 num_hiddens=NUM_HIDDENS, num_residual_layers=NUM_RESIDUAL_LAYERS, commitment_cost=COMMITMENT_COST,
                 decay=DECAY):
        super(VQVAE, self).__init__()

        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, embedding_dim)
        self._pre_vq_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, stride=1)
        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, in_channels)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        vq_loss, quantized, perplexity, encoding_indices = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return x_recon, vq_loss, quantized, encoding_indices, z, perplexity

    def initialize_codebook(self, data_loader, device, num_batches_for_init=50):
        """
        Gathers encoder outputs and initializes the codebook using K-Means.
        This method should be called once before starting the training loop.
        """
        self.to(device)
        self.eval()  # Set model to evaluation mode

        all_encoder_outputs = []
        print(f"Gathering encoder outputs from {num_batches_for_init} batches for K-Means initialization...")
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if i >= num_batches_for_init:
                    break
                data = data.to(device)
                z = self._encoder(data)
                z = self._pre_vq_conv(z)
                # z shape: (B, C, H, W) -> flatten to (B*H*W, C)
                all_encoder_outputs.append(z.permute(0, 2, 3, 1).reshape(-1, VQVAE_EMBEDDING_DIM))

        # Pass the collected encoder outputs to the quantizer's init method
        self._vq_vae.initialize_codebook_with_kmeans(torch.cat(all_encoder_outputs, dim=0))
        self.train()  # Set model back to training mode


# --- Example Usage ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model
    model = VQVAE().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- K-Means Initialization Step ---
    # Create a dummy DataLoader to simulate a real dataset
    dummy_dataset = TensorDataset(torch.randn(256, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), torch.zeros(256))
    dummy_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    # Call the initialization method before starting training
    model.initialize_codebook(dummy_loader, device, num_batches_for_init=4)

    # --- Regular Forward Pass (as in a training loop) ---
    # Create a dummy input tensor
    dummy_input = torch.randn(4, IMG_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)

    # Forward pass
    model.train()  # Ensure model is in training mode for EMA updates
    x_recon, vq_loss, quantized, encoding_indices, z, perplexity = model(dummy_input)

    # Print shapes and values to verify
    print("\n--- Output Verification after K-Means Init ---")
    print(f"Input shape:          {dummy_input.shape}")
    print(f"Encoder output (z) shape: {z.shape}")
    print(f"Quantized shape:      {quantized.shape}")
    print(f"Encoding indices shape: {encoding_indices.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"VQ Loss:              {vq_loss.item():.4f}")
    print(f"Perplexity:           {perplexity.item():.4f}")
