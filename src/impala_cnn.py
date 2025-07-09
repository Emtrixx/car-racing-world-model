import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Constants for the ImpalaCNN architecture
IN_CHANNELS = 1  # Input channels for the image (grayscale)
IMG_SIZE = 64  # Input image size (height and width)
EMBEDDING_DIM = 1024  # The output dimension of the ImpalaCNN


class ResidualBlock(nn.Module):
    """
    The ResNet block as described in the paper.
    Each block is composed of (a) a ReLU activation, (b) an instance normalization,
    and (c) a convolutional layer with kernel size 3x3 and stride of 1.
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.inst_norm = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.inst_norm(out)
        out = self.conv(out)
        out += residual  # Add the residual connection
        return out


class ImpalaCNN(nn.Module):
    """
    A smaller ImpalaCNN architecture.
    """

    def __init__(self, in_channels=IN_CHANNELS, img_size=IMG_SIZE, out_dim=EMBEDDING_DIM):
        super(ImpalaCNN, self).__init__()

        # Define the three main stacks
        stacks = []
        stack_channels = [32, 64, 64]
        current_channels = in_channels

        for channels in stack_channels:
            stack = nn.Sequential(
                nn.InstanceNorm2d(current_channels),
                nn.Conv2d(current_channels, channels, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ResidualBlock(channels),
                ResidualBlock(channels)
            )
            stacks.append(stack)
            current_channels = channels

        self.stacks = nn.ModuleList(stacks)
        self.final_relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Determine the flattened size after the smaller convolutional stacks
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_size, img_size)
            x = dummy_input
            for stack in self.stacks:
                x = stack(x)
            x = self.final_relu(x)
            x = self.flatten(x)
            # New flattened size will be 64 * 8 * 8 = 4096
            flattened_size = x.shape[1]

        # Final linear layer to project from the new flattened size to the desired output
        self.final_linear = nn.Linear(flattened_size, out_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 63, 63).
        Returns:
            torch.Tensor: Flattened embedding vector of shape (batch_size, 1024).
        """
        for stack in self.stacks:
            x = stack(x)
        x = self.final_relu(x)
        x = self.flatten(x)
        x = self.final_linear(x)
        return x


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for SB3.
    It handles the observation format and forwards it to the ImpalaCNN.
    """

    def __init__(self, observation_space, features_dim: int = 1024):
        super().__init__(observation_space, features_dim)

        # SB3 observation shape: (num_stack, height, width, channels)
        num_stack = observation_space.shape[0]
        # Assuming the base env has 1 channel after preprocessing
        original_channels = observation_space.shape[-1]
        in_channels = num_stack * original_channels

        self.cnn = ImpalaCNN(in_channels=in_channels, out_dim=features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 observations for images are (batch_size, height, width, channels).
        # With FrameStack, it becomes (batch_size, num_stack, H, W, C).
        # We need to convert it to what PyTorch CNN expects: (batch_size, in_channels, H, W).

        # Permute to (batch_size, H, W, num_stack, C)
        permuted_obs = observations.permute(0, 2, 3, 1, 4)

        # Reshape to combine stack and channel dimensions: (batch_size, H, W, num_stack * C)
        batch_size = permuted_obs.shape[0]
        h, w = permuted_obs.shape[1], permuted_obs.shape[2]
        combined_channels = permuted_obs.shape[3] * permuted_obs.shape[4]
        reshaped_obs = permuted_obs.reshape(batch_size, h, w, combined_channels)

        # Permute to PyTorch "channels-first" format: (batch_size, num_stack * C, H, W)
        final_obs = reshaped_obs.permute(0, 3, 1, 2)

        return self.cnn(final_obs)


# Example usage:
if __name__ == '__main__':
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_image = torch.randn(4, IN_CHANNELS, IMG_SIZE, IMG_SIZE)

    # Instantiate the smaller model
    impala_cnn = ImpalaCNN(in_channels=IN_CHANNELS, out_dim=EMBEDDING_DIM)
    print("Smaller ImpalaCNN Model Architecture:")
    print(impala_cnn)

    # Get the output embedding
    embedding = impala_cnn(dummy_image)

    # Print the output shape
    print(f"\nInput shape: {dummy_image.shape}")
    print(f"Output embedding shape: {embedding.shape}")
