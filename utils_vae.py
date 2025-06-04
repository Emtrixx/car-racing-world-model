import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from torch.utils.data import Dataset

from conv_vae import ConvVAE
from utils import DEVICE, VAE_CHECKPOINT_FILENAME, NUM_STACK, transform, LATENT_DIM, make_env_sb3

SB3_MODEL_FILENAME = f"sb3_default_carracing-v3_best/best_model.zip"  # best
SB3_MODEL_PATH = pathlib.Path("checkpoints") / SB3_MODEL_FILENAME


# --- Data Collection --- (uses vae_model and PPO actor to collect frames)
def collect_frames(env_name, num_frames, transform_fn):
    print(f"Collecting {num_frames} frames for VAE training...")

    # Load Pre-trained VAE
    vae_model = ConvVAE().to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found. Train VAE first.")
        exit()
    except Exception as e:
        print(f"ERROR loading VAE: {e}");
        exit()

    # --- Create Environment using make_env_sb3 ---
    # make_env_sb3 handles all necessary wrappers including LatentStateWrapper and ActionTransformWrapper
    # It needs the VAE instance.
    # For playback, gamma for NormalizeReward wrapper doesn't strictly matter but use a sensible default.
    try:
        env = make_env_sb3(
            env_id=env_name,
            vae_model_instance=vae_model,
            transform_function=transform,
            frame_stack_num=NUM_STACK,
            single_latent_dim=LATENT_DIM,
            device_for_vae=DEVICE,
            gamma=0.99,  # Standard gamma, used by NormalizeReward
            max_episode_steps=1000,  # Typical for CarRacing
            render_mode="rgb_array",  # Use rgb_array for frame collection
            seed=np.random.randint(0, 10000)  # Give a random seed for variety if desired
        )
        print("Environment created successfully with make_env_sb3.")
    except Exception as e:
        print(f"Error creating environment with make_env_sb3: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load Trained SB3 PPO Agent ---
    print(f"Loading trained SB3 PPO agent from: {SB3_MODEL_PATH}")
    if not SB3_MODEL_PATH.exists():
        print(f"ERROR: SB3 PPO Model not found at {SB3_MODEL_PATH}")
        if hasattr(env, 'close'): env.close()
        return
    try:
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=env)  # Provide env for action/obs space checks
        print(f"Successfully loaded SB3 PPO agent. Agent device: {ppo_agent.device}")
    except Exception as e:
        print(f"ERROR loading SB3 PPO agent: {e}")
        if hasattr(env, 'close'): env.close()
        import traceback
        traceback.print_exc()
        return

    frames = []
    Z_t_numpy, _ = env.reset()
    frame_count = 0

    while frame_count < num_frames:

        action, _state = ppo_agent.predict(Z_t_numpy)

        Z_t_numpy, reward, terminated, truncated, info = env.step(action)
        frame = env.render()

        if frame is not None:
            processed_frame = transform_fn(frame)  # Use transform from utils
            frames.append(processed_frame)
            frame_count += 1

        if terminated or truncated:
            Z_t_numpy, _ = env.reset()

        if frame_count % 500 == 0 and frame_count > 0:
            print(f"  Collected {frame_count}/{num_frames} frames...")

    env.close()
    print(f"Finished collecting {len(frames)} frames.")
    return torch.stack(frames)


# --- Dataset Class ---
class FrameDataset(Dataset):
    def __init__(self, frame_data):
        self.data = frame_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Visualization ---
def visualize_reconstruction(model, dataloader, device, epoch, n_samples=8):
    model.eval()
    data = next(iter(dataloader)).to(device)
    if data.size(0) > n_samples: data = data[:n_samples]

    with torch.no_grad():
        recon_batch, _, _ = model(data)

    original = data.cpu()
    reconstructed = recon_batch.cpu()

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    fig.suptitle(f'Epoch {epoch} - Original vs. Reconstructed', fontsize=16)
    for i in range(n_samples):
        img_orig = original[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(np.clip(img_orig, 0, 1))
        axes[0, i].set_title(f'Original {i + 1}')
        axes[0, i].axis('off')
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        axes[1, i].imshow(np.clip(img_recon, 0, 1))
        axes[1, i].set_title(f'Recon {i + 1}')
        axes[1, i].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"images/vqvae_reconstruction_epoch_{epoch}.png"
    plt.savefig(save_path)
    print(f"Saved reconstruction visualization to {save_path}")
    plt.close(fig)  # Close the figure to free memory
