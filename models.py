# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import necessary constants from utils
from utils import CHANNELS, LATENT_DIM, IMG_SIZE, ACTION_DIM, WM_HIDDEN_DIM

# --- VAE Model Definition ---
class ConvVAE(nn.Module):
    def __init__(self, img_channels=CHANNELS, latent_dim=LATENT_DIM, img_size=IMG_SIZE):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Use a fixed conv_output_size known from the architecture or calculate dynamically
        # For IMG_SIZE=64, the output is 256*4*4 = 4096
        self.conv_output_size = 256 * (img_size // 16) * (img_size // 16) # Generalize calculation

        # --- Encoder ---
        self.enc_conv1 = nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1) # -> /2
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # -> /4
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)          # -> /8
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)         # -> /16
        self.enc_fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.enc_fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        # --- Decoder ---
        self.dec_fc = nn.Linear(latent_dim, self.conv_output_size)
        # Calculate the H/W dimension before the first transpose conv
        self.dec_start_h_w = img_size // 16
        self.dec_convT1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # -> *2
        self.dec_convT2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # -> *4
        self.dec_convT3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # -> *8
        self.dec_convT4 = nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1) # -> *16 (original size)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1) # Flatten all dimensions except batch
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample epsilon from standard normal distribution
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        # Reshape to match the input shape of the first ConvTranspose2d layer -> (B, 256, H/16, W/16)
        x = x.view(-1, 256, self.dec_start_h_w, self.dec_start_h_w)
        x = F.relu(self.dec_convT1(x))
        x = F.relu(self.dec_convT2(x))
        x = F.relu(self.dec_convT3(x))
        # Apply sigmoid to the output layer to constrain pixels to [0, 1]
        reconstruction = torch.sigmoid(self.dec_convT4(x))
        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# --- World Model Definition ---
class WorldModelMLP(nn.Module):
    # Use WM_HIDDEN_DIM from utils or define a default
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_z_pred = nn.Linear(hidden_dim, latent_dim)
        # Optional heads can be added later
        # self.fc_r_pred = nn.Linear(hidden_dim, 1)

    def forward(self, z, a):
        # Ensure a is float tensor and on same device as z
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32, device=z.device)
        elif a.dtype != torch.float32:
            a = a.float()
        a = a.to(z.device)

        # Handle batch dimension if missing (e.g., during single step prediction)
        if z.ndim == 1: z = z.unsqueeze(0)
        if a.ndim == 1: a = a.unsqueeze(0)

        za = torch.cat([z, a], dim=-1)
        hidden = F.relu(self.fc1(za))
        hidden = F.relu(self.fc2(hidden))
        next_z_pred = self.fc_z_pred(hidden)
        return next_z_pred