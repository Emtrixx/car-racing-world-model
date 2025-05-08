# utils.py
import torch
import numpy as np
import gymnasium as gym
from torchvision import transforms
from torch.distributions import Normal

# --- Configuration Constants ---
ENV_NAME = "CarRacing-v3"
WM_HIDDEN_DIM = 256 # Hidden dimension for the World Model MLP
IMG_SIZE = 64       # Resize frames
CHANNELS = 3       # RGB channels
LATENT_DIM = 32     # Size of the latent space vector z
ACTION_DIM = 3      # CarRacing: Steering, Gas, Brake

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- File Paths ---
# It's often good practice to make these easily configurable (e.g., via argparse)
# But defining them here keeps consistency for now.
VAE_MODEL_SUFFIX = f"ld{LATENT_DIM}" # Suffix based on latent dim
VAE_CHECKPOINT_FILENAME = f"checkpoints/{ENV_NAME}_cvae_ld{LATENT_DIM}_epoch10.pth" # Your saved VAE model
WM_MODEL_SUFFIX = f"ld{LATENT_DIM}_ac{ACTION_DIM}"
WM_CHECKPOINT_FILENAME = f"checkpoints/{ENV_NAME}_worldmodel_mlp_{WM_MODEL_SUFFIX}.pth"
WM_CHECKPOINT_FILENAME_GRU = f"checkpoints/{ENV_NAME}_worldmodel_gru_{WM_MODEL_SUFFIX}.pth"
PPO_ACTOR_SAVE_FILENAME = f"checkpoints/{ENV_NAME}_ppo_actor_ld{LATENT_DIM}.pth"
PPO_CRITIC_SAVE_FILENAME = f"checkpoints/{ENV_NAME}_ppo_critic_ld{LATENT_DIM}.pth"

# --- Data Preprocessing Transform (Identical for all parts) ---
transform = transforms.Compose([
    transforms.ToPILImage(),          # Convert numpy array to PIL Image
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),            # Convert PIL Image to tensor (C, H, W) and scales to [0, 1]
])

# --- Simple Random Policy ---
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state): # state can be observation or latent state, ignored here
        return self.action_space.sample()

class PPOPolicyWrapper:
    def __init__(self, actor_model, device, deterministic=False, action_space_low=None, action_space_high=None):
        self.actor_model = actor_model
        self.device = device
        self.deterministic = deterministic # False for exploration, True for exploitation/eval

        if action_space_low is None:
            # Defaults for CarRacing-v3
            action_space_low = [-1.0, 0.0, 0.0]
        if action_space_high is None:
            action_space_high = [1.0, 1.0, 1.0]

        self.action_space_low_tensor = torch.tensor(action_space_low, device=self.device, dtype=torch.float32)
        self.action_space_high_tensor = torch.tensor(action_space_high, device=self.device, dtype=torch.float32)


    def get_action(self, z_t_numpy): # Expects a numpy array for z_t
        self.actor_model.eval() # Ensure actor is in eval mode
        z_t = torch.tensor(z_t_numpy, dtype=torch.float32).to(self.device).unsqueeze(0) # Add batch dim

        with torch.no_grad():
            dist = self.actor_model(z_t) # actor_model should be the loaded Actor network
            if self.deterministic:
                action_raw = dist.mean
            else:
                action_raw = dist.sample() # Sample for exploration

            # Process action (same logic as in train_ppo.py and play_game.py)
            # Steering is output by actor's fc_mean in tanh range already for its first component
            # Gas/Brake means are unbounded from fc_mean, then dist samples.
            # We apply tanh to the sample, then scale.

            action_processed = torch.tanh(action_raw) # Squash sample to [-1, 1]

            action_scaled = torch.zeros_like(action_processed)
            action_scaled[:, 0] = action_processed[:, 0] # Steering: directly use tanh output
            action_scaled[:, 1:] = (action_processed[:, 1:] + 1.0) / 2.0 # Gas, Brake: scale from [-1,1] to [0,1]

            action_clipped = torch.clamp(action_scaled,
                                         self.action_space_low_tensor,
                                         self.action_space_high_tensor)

        return action_clipped.squeeze(0).cpu().numpy()

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
    processed_obs = transform_fn(obs).unsqueeze(0).to(device) # Add batch dim and move to device
    with torch.no_grad(): # We don't need gradients for VAE encoding
        mu, logvar = vae_model.encode(processed_obs)
        # Using the mean (mu) is common for downstream tasks
        z = mu # Shape: (1, LATENT_DIM)
    return z.squeeze(0) # Remove batch dim -> Shape: (LATENT_DIM)

# You could add other shared utilities here, e.g., functions to save/load models,
# setup logging, etc.

print(f"Utils loaded. Using device: {DEVICE}")
print(f"VAE Path: {VAE_CHECKPOINT_FILENAME}")
print(f"WM Path: {WM_CHECKPOINT_FILENAME}")
PPO_DREAM_ACTOR_SAVE_FILENAME = f"{ENV_NAME}_ppo_dream_actor_ld{LATENT_DIM}.pth"
PPO_DREAM_CRITIC_SAVE_FILENAME = f"{ENV_NAME}_ppo_dream_critic_ld{LATENT_DIM}.pth"
