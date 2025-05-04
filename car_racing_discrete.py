import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset # Optional but good practice
import matplotlib.pyplot as plt
import random
import time
from collections import deque # Useful for replay buffer
import sys

# --- Configuration ---
ENV_NAME = "CarRacing-v3"

# VAE Config (Must match the saved model)
IMG_SIZE = 64
CHANNELS = 3
LATENT_DIM = 32
VAE_CHECKPOINT_PATH = f"../{ENV_NAME}_cvae_ld{LATENT_DIM}_epoch10.pth" # Your saved VAE model

# World Model & Training Config
ACTION_DIM = 3  # CarRacing: Steering, Gas, Brake
WM_HIDDEN_DIM = 256
WM_LEARNING_RATE = 1e-4
WM_EPOCHS = 20 # Epochs to train the world model
WM_BATCH_SIZE = 64
COLLECT_STEPS = 10000 # Number of environment steps to collect for WM training
REPLAY_BUFFER_CAPACITY = COLLECT_STEPS # Store all collected transitions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Loading VAE from: {VAE_CHECKPOINT_PATH}")
print(f"World Model Latent Dim: {LATENT_DIM}, Action Dim: {ACTION_DIM}")

# --- Data Preprocessing (Must be IDENTICAL to VAE training) ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# --- VAE Model Definition (Must match the saved model architecture) ---
class ConvVAE(nn.Module):
    def __init__(self, img_channels=CHANNELS, latent_dim=LATENT_DIM, img_size=IMG_SIZE):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self._determine_conv_output_size(img_channels, img_size)

        self.enc_conv1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc_fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.enc_fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, self.conv_output_size)
        self.dec_convT1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_convT2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_convT3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_convT4 = nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1)

    def _determine_conv_output_size(self, img_channels, img_size):
        dummy_conv1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)
        dummy_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        dummy_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        dummy_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        x = torch.zeros(1, img_channels, img_size, img_size)
        x = F.relu(dummy_conv1(x))
        x = F.relu(dummy_conv2(x))
        x = F.relu(dummy_conv3(x))
        x = F.relu(dummy_conv4(x))
        self.conv_output_size = int(np.prod(x.size()[1:]))
        # print(f"Determined flattened conv output size: {self.conv_output_size}") # Should be 4096

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    # We only need the encoder part for world model training, but keep full VAE for structure
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        h_w = int(np.sqrt(self.conv_output_size / 256))
        x = x.view(-1, 256, h_w, h_w)
        x = F.relu(self.dec_convT1(x))
        x = F.relu(self.dec_convT2(x))
        x = F.relu(self.dec_convT3(x))
        reconstruction = torch.sigmoid(self.dec_convT4(x))
        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# --- World Model Definition ---
class WorldModelMLP(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=WM_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_z_pred = nn.Linear(hidden_dim, latent_dim)
        # Add optional reward/done prediction heads later if needed
        # self.fc_r_pred = nn.Linear(hidden_dim, 1)
        # self.fc_d_pred = nn.Linear(hidden_dim, 1)

    def forward(self, z, a):
        # Ensure a is float tensor
        if not isinstance(a, torch.Tensor):
             a = torch.tensor(a, dtype=torch.float32, device=z.device)
        elif a.dtype != torch.float32:
             a = a.float()

        # Ensure z and a are on the same device and have batch dim
        a = a.to(z.device)
        if len(a.shape) == 1: # Add batch dim if missing
             a = a.unsqueeze(0)
        if len(z.shape) == 1:
             z = z.unsqueeze(0)

        # Concatenate latent state and action
        za = torch.cat([z, a], dim=-1)
        hidden = F.relu(self.fc1(za))
        hidden = F.relu(self.fc2(hidden))
        next_z_pred = self.fc_z_pred(hidden)
        # next_r_pred = self.fc_r_pred(hidden) # Optional
        # next_d_pred = torch.sigmoid(self.fc_d_pred(hidden)) # Optional (use sigmoid for done prob)
        return next_z_pred #, next_r_pred, next_d_pred

# --- Policy Definition ---
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state): # state can be observation or latent state, ignored here
        return self.action_space.sample()

# --- Helper Function to Preprocess and Encode ---
def preprocess_and_encode(obs, transform, vae_model, device):
    """Applies transform and encodes observation using the VAE encoder."""
    processed_obs = transform(obs).unsqueeze(0).to(device) # Add batch dim and move to device
    with torch.no_grad(): # We don't need gradients for VAE encoding
        mu, logvar = vae_model.encode(processed_obs)
        # For the world model, using the mean (mu) is often sufficient and simpler
        z = mu # Shape: (1, LATENT_DIM)
    return z.squeeze(0) # Remove batch dim for storage -> Shape: (LATENT_DIM)

# --- Data Collection with Latent States ---
def collect_latent_transitions(env, policy, transform, vae_model, replay_buffer, num_steps, device):
    print(f"Collecting {num_steps} transitions using the policy...")
    obs, _ = env.reset()
    z_current = preprocess_and_encode(obs, transform, vae_model, device)
    collected_count = 0

    while collected_count < num_steps:
        # 1. Get action from policy (using current latent state z_current)
        #    Note: Random policy ignores the state, but a real RL policy would use z_current
        action = policy.get_action(z_current.cpu().numpy()) # Policy might expect numpy

        # 2. Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        collected_count += 1

        # 3. Preprocess and encode the next observation
        z_next = preprocess_and_encode(next_obs, transform, vae_model, device)

        # 4. Store the transition (z_t, a_t, r_{t+1}, z_{t+1}, done)
        #    Convert action to tensor for storage consistency
        action_tensor = torch.tensor(action, dtype=torch.float32)
        transition = (z_current.cpu(), action_tensor, torch.tensor(reward, dtype=torch.float32), z_next.cpu(), torch.tensor(done, dtype=torch.bool))
        replay_buffer.append(transition)

        # 5. Update current state for next iteration
        z_current = z_next
        if done:
            obs, _ = env.reset()
            z_current = preprocess_and_encode(obs, transform, vae_model, device)

        if collected_count % 1000 == 0:
            print(f"  Collected {collected_count}/{num_steps} transitions...")

    print(f"Finished collecting {len(replay_buffer)} transitions.")
    return replay_buffer


# --- Dataset for World Model Training ---
class TransitionDataset(Dataset):
    def __init__(self, buffer):
        # Convert list of tuples into tuples of lists/tensors
        self.data = list(buffer) # Make a copy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        z_t, a_t, r_tp1, z_tp1, done = self.data[idx]
        # Return tensors needed for world model training
        return z_t, a_t, z_tp1

# --- World Model Training Loop ---
def train_world_model(world_model, dataloader, optimizer, criterion, epochs, device):
    print("Training World Model...")
    world_model.train() # Set model to training mode
    losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        processed_batches = 0
        for batch_idx, (z_t, a_t, z_tp1) in enumerate(dataloader):
            z_t, a_t, z_tp1 = z_t.to(device), a_t.to(device), z_tp1.to(device)

            optimizer.zero_grad()

            # Get prediction
            z_tp1_pred = world_model(z_t, a_t)

            # Calculate loss (predicting next latent state)
            loss = criterion(z_tp1_pred, z_tp1)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            processed_batches += 1

            # Optional: Print batch progress
            # if batch_idx % 100 == 0:
            #     print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / processed_batches
        losses.append(avg_epoch_loss)
        print(f'====> World Model Epoch: {epoch} Average loss: {avg_epoch_loss:.6f}')

    world_model.eval() # Set back to evaluation mode
    print("Finished training World Model.")
    return losses


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize Environment
    # Use wrapper for grayscale and frame stacking if needed later, but VAE uses RGB now
    env = gym.make(ENV_NAME, render_mode="rgb_array") # Render mode needed for VAE
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}") # Should be (96, 96, 3) originally

    # 2. Load Pre-trained VAE
    vae_model = ConvVAE(img_channels=CHANNELS, latent_dim=LATENT_DIM, img_size=IMG_SIZE).to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_PATH, map_location=DEVICE))
        vae_model.eval() # Set VAE to evaluation mode
        print(f"Successfully loaded VAE weights from {VAE_CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint not found at {VAE_CHECKPOINT_PATH}. Please train VAE first.")
        sys.exit()
    except Exception as e:
        print(f"ERROR: Failed to load VAE weights: {e}")
        sys.exit()

    # 3. Initialize Policy (Random for now)
    policy = RandomPolicy(env.action_space)

    # 4. Initialize Replay Buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_CAPACITY)

    # 5. Collect Data using VAE Encoder and Policy
    start_collect_time = time.time()
    replay_buffer = collect_latent_transitions(env, policy, transform, vae_model, replay_buffer, COLLECT_STEPS, DEVICE)
    env.close() # Close env after collection
    print(f"Data collection took {time.time() - start_collect_time:.2f} seconds.")

    # 6. Prepare DataLoader for World Model Training
    transition_dataset = TransitionDataset(replay_buffer)
    wm_dataloader = DataLoader(transition_dataset, batch_size=WM_BATCH_SIZE, shuffle=True)

    # 7. Initialize World Model and Optimizer
    world_model = WorldModelMLP(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=WM_HIDDEN_DIM).to(DEVICE)
    wm_optimizer = optim.Adam(world_model.parameters(), lr=WM_LEARNING_RATE)
    wm_criterion = nn.MSELoss() # Mean Squared Error for predicting next latent state

    # 8. Train the World Model
    start_train_time = time.time()
    wm_losses = train_world_model(world_model, wm_dataloader, wm_optimizer, wm_criterion, WM_EPOCHS, DEVICE)
    print(f"World Model training took {time.time() - start_train_time:.2f} seconds.")

    # Optional: Plot world model training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, WM_EPOCHS + 1), wm_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("World Model Training Loss (Predicting $z_{t+1}$)")
    plt.grid(True)
    plt.savefig("world_model_training_loss.png")
    print("Saved world model training loss plot to world_model_training_loss.png")
    # plt.show()

    # 9. Save the trained World Model (Optional)
    try:
        wm_save_path = f"{ENV_NAME}_worldmodel_mlp_ld{LATENT_DIM}_ac{ACTION_DIM}.pth"
        torch.save(world_model.state_dict(), wm_save_path)
        print(f"World Model saved to {wm_save_path}")
    except Exception as e:
        print(f"Error saving World Model: {e}")

    # --- Next Steps ---
    # Now you have a trained world model. You can use it for:
    # 1. Evaluating its prediction accuracy on new data.
    # 2. Implementing a planning algorithm (like CEM or MPC) that uses the world model.
    # 3. Training an RL Controller (replacing RandomPolicy) using imagined rollouts generated by the world model (e.g., with CMA-ES or Dreamer-like approaches).
    # 4. Iteratively collecting more data with a better policy and retraining the world model.