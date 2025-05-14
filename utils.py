# utils.py
import torch
from torchvision import transforms
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import numpy as np

# --- Configuration Constants ---
ENV_NAME = "CarRacing-v3"
WM_HIDDEN_DIM = 256  # Hidden dimension for the World Model MLP
IMG_SIZE = 64  # Resize frames
CHANNELS = 3  # RGB channels
NUM_STACK = 4  # Number of latent vectors to stack
LATENT_DIM = 32  # Size of the latent space vector z
ACTION_DIM = 3  # CarRacing: Steering, Gas, Brake

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- File Paths ---
# It's often good practice to make these easily configurable (e.g., via argparse)
# But defining them here keeps consistency for now.
VAE_MODEL_SUFFIX = f"ld{LATENT_DIM}"  # Suffix based on latent dim
VAE_CHECKPOINT_FILENAME = f"checkpoints/{ENV_NAME}_cvae_ld{LATENT_DIM}_epoch10.pth"  # Your saved VAE model
WM_MODEL_SUFFIX = f"ld{LATENT_DIM}_ac{ACTION_DIM}"
WM_CHECKPOINT_FILENAME = f"checkpoints/{ENV_NAME}_worldmodel_mlp_{WM_MODEL_SUFFIX}.pth"
WM_CHECKPOINT_FILENAME_GRU = f"checkpoints/{ENV_NAME}_worldmodel_gru_{WM_MODEL_SUFFIX}.pth"

# --- Data Preprocessing Transform (Identical for all parts) ---
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # Convert PIL Image to tensor (C, H, W) and scales to [0, 1]
])


# --- Helper Function to Preprocess and Encode Observation ---
def preprocess_and_encode(obs, transform_fn, vae_model, device):
    """
    Applies transform and encodes observation using the VAE encoder's mean.

    Args:
        obs (np.array): Raw observation from environment.
        transform_fn (callable): The preprocessing transform.
        vae_model (nn.Module): The loaded VAE model (in eval mode).
        device (torch.device): The target device.

    Returns:
        torch.Tensor: Latent state vector z (mean) on the specified device.
                      Shape: (LATENT_DIM)
    """
    processed_obs = transform_fn(obs).unsqueeze(0).to(device)  # Add batch dim and move to device
    with torch.no_grad():  # We don't need gradients for VAE encoding
        mu, logvar = vae_model.encode(processed_obs)
        # Using the mean (mu) is common for downstream tasks
        z = mu  # Shape: (1, LATENT_DIM)
    return z.squeeze(0)  # Remove batch dim -> Shape: (LATENT_DIM)


# --- Helper Function to Preprocess and Encode Observation Stack ---
def preprocess_and_encode_stack(
        raw_frame_stack,  # NumPy array from FrameStackWrapper: (num_stack, H, W, C)
        transform_fn,  # Your existing torchvision transform
        vae_model,  # Your loaded VAE model (in eval mode)
        device
):
    latent_vectors = []
    for i in range(raw_frame_stack.shape[0]):  # Iterate through N frames in the stack
        raw_frame = raw_frame_stack[i]  # Single frame (H, W, C)
        # Use your existing preprocess_and_encode logic for a single frame,
        # or replicate its core here:
        processed_frame = transform_fn(raw_frame).unsqueeze(0).to(device)  # (1, C, H, W)
        with torch.no_grad():
            mu, _ = vae_model.encode(processed_frame)  # mu shape (1, LATENT_DIM)
            latent_vectors.append(mu.squeeze(0))  # Squeeze to (LATENT_DIM)

    # Concatenate the N latent vectors
    # Resulting shape: (num_stack * LATENT_DIM,)
    concatenated_latents = torch.cat(latent_vectors, dim=0)
    return concatenated_latents


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=NUM_STACK):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Modify observation space
        # Original observation space is Box(0, 255, (H, W, C), uint8)
        # We will stack raw frames. The VAE will process them individually later.
        # example carlculation. observation_space defined with same result below
        low = np.repeat(self.observation_space.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(self.observation_space.high[..., np.newaxis], num_stack, axis=-1)
        # New shape: (H, W, C*num_stack) if concatenating channels, or (H, W, C, num_stack)
        # Let's make it (num_stack, H, W, C) for easier iteration later.
        # Or if VAE expects (C,H,W), then (num_stack, C, H, W) after transform.
        # For now, the wrapper will output a list of N frames, or a NumPy array (N, H, W, C).
        # The actual stacking for VAE input will be handled after this wrapper.
        # So, the wrapper's observation space can be a tuple of spaces or a Box with num_stack as first dim.
        # Let's return a list of frames from the wrapper for maximum flexibility.
        # However, a NumPy array is more standard for gym observation_space.
        original_shape = self.env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,  # Assuming raw frames are uint8 [0,255]
            high=255,
            shape=(num_stack, *original_shape),  # (num_stack, H, W, C)
            dtype=self.env.observation_space.dtype
        )

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, "Not enough frames in buffer"
        return np.array(list(self.frames), dtype=self.env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)  # Repeat first frame N times
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info


print(f"Utils loaded. Using device: {DEVICE}")
print(f"VAE Path: {VAE_CHECKPOINT_FILENAME}")
print(f"WM Path: {WM_CHECKPOINT_FILENAME}")
