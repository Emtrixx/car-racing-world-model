import argparse
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from play_game_sb3 import SB3_MODEL_PATH

from src.transformer_world_model import WorldModelTransformer, HISTORY_LEN
from src.utils import (DEVICE, VQ_VAE_CHECKPOINT_FILENAME, ENV_NAME,
                       WM_CHECKPOINT_FILENAME_TRANSFORMER, make_env_sb3, NUM_STACK, IMAGES_DIR, ACTION_DIM)
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM
from src.train_dyna_loop import get_vq_indices  # Assuming this function is correct


def collect_real_buffer(env, vq_vae, ppo_agent, buffer_size, device):
    """Collects a buffer of real experience to initialize the dream environment."""
    print(f"Collecting a buffer of {buffer_size} real transitions...")
    real_buffer = deque(maxlen=buffer_size)
    obs, _ = env.reset()
    is_first_step = True

    for _ in tqdm(range(buffer_size), desc="Collecting real experience"):
        # obs[-1] gets the latest frame from the stack
        prev_tokens = get_vq_indices(vq_vae, obs[-1], device)
        action, _ = ppo_agent.predict(obs, deterministic=False)
        next_obs, reward, done, truncated, info = env.step(action)
        next_tokens = get_vq_indices(vq_vae, next_obs[-1], device)

        real_buffer.append({
            "prev_tokens": prev_tokens.squeeze(0).cpu(),
            "action": torch.from_numpy(action).float(),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([done or truncated], dtype=torch.float32),
            "next_tokens": next_tokens.squeeze(0).cpu(),
            "is_first_step": torch.tensor([is_first_step], dtype=torch.bool)
        })

        obs = next_obs
        is_first_step = False
        if done or truncated:
            obs, _ = env.reset()
            is_first_step = True

    return list(real_buffer)


def draw_attention_maps(model_path, vq_vae_path, ppo_path, output_dir):
    """
    Loads models, collects real data, generates attention maps, and saves them as plots.
    """
    print(f"Using device: {DEVICE}")

    # --- Configuration ---
    REAL_BUFFER_SIZE = 500

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # --- Load Models ---
    print("Loading models (VQ-VAE, PPO Agent, and World Model)...")
    real_env = make_env_sb3(env_id=ENV_NAME, frame_stack_num=NUM_STACK)

    vq_vae = VQVAE().to(DEVICE)
    vq_vae.load_state_dict(torch.load(vq_vae_path, map_location=DEVICE))
    vq_vae.eval()

    ppo_agent = PPO.load(ppo_path, device=DEVICE, env=real_env)

    # Note: The model architecture must match the one used for training.
    world_model = WorldModelTransformer(
        vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
    ).to(DEVICE)
    world_model = torch.compile(world_model)
    world_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    world_model.eval()
    print("All models loaded successfully.")

    # --- Collect Real Experience ---
    real_experience_buffer = collect_real_buffer(real_env, vq_vae, ppo_agent, REAL_BUFFER_SIZE, DEVICE)
    real_env.close()

    # --- Prepare History from Real Buffer ---
    print("Preparing realistic history from collected data...")
    # Find valid start indices
    valid_start_indices = []
    for i in range(len(real_experience_buffer) - HISTORY_LEN):
        is_valid = True
        for j in range(i + 1, i + HISTORY_LEN):
            if real_experience_buffer[j]['is_first_step'] or real_experience_buffer[j - 1]['done']:
                is_valid = False
                break
        if is_valid:
            valid_start_indices.append(i)

    if not valid_start_indices:
        raise ValueError("No valid start indices found in the replay buffer.")

    start_idx = random.choice(valid_start_indices)

    action_history = torch.stack([real_experience_buffer[start_idx + i]['action'] for i in range(HISTORY_LEN)])
    token_history = torch.stack([real_experience_buffer[start_idx + i]['prev_tokens'] for i in range(HISTORY_LEN)])

    action_hist_tensor = action_history.to(DEVICE).unsqueeze(0)
    token_hist_tensor = token_history.to(DEVICE).unsqueeze(0)

    # --- Get Attention Maps ---
    print("Generating attention maps...")
    with torch.no_grad():
        _, _, _, attention_maps = world_model.generate(
            action_hist_tensor, token_hist_tensor, get_attention=True
        )

    if not attention_maps:
        print("ERROR: Could not retrieve attention maps.")
        return

    print(
        f"Successfully retrieved attention maps for {len(attention_maps.get('cross', []))} cross-attention and {len(attention_maps.get('self', []))} self-attention layers.")

    # --- Plot and Save Maps ---
    num_heads = 0
    if 'cross' in attention_maps and attention_maps['cross']:
        num_heads = attention_maps['cross'][0].shape[1]
    elif 'self' in attention_maps and attention_maps['self']:
        num_heads = attention_maps['self'][0].shape[1]

    if num_heads == 0:
        print("Warning: No attention maps found to plot.")
        return

    for i in range(world_model.transformer_decoder.num_layers):
        print(f"Plotting maps for Layer {i}...")
        for h in range(num_heads):
            # Cross-Attention
            if 'cross' in attention_maps and i < len(attention_maps['cross']):
                fig, ax = plt.subplots(figsize=(12, 6))
                attn_map = attention_maps['cross'][i][0, h].cpu().numpy()
                im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
                ax.set_title(f"Cross-Attention: Layer {i}, Head {h}")
                ax.set_xlabel("Key: State & Action History Tokens")
                ax.set_ylabel("Query: Next State Tokens")
                fig.colorbar(im, ax=ax)
                plt.tight_layout()
                plot_filename = Path(output_dir) / f"cross_attention_layer_{i}_head_{h}.png"
                plt.savefig(plot_filename)
                plt.close(fig)

            # Self-Attention
            if 'self' in attention_maps and i < len(attention_maps['self']):
                fig, ax = plt.subplots(figsize=(10, 10))
                attn_map_self = attention_maps['self'][i][0, h].cpu().numpy()
                im = ax.imshow(attn_map_self, cmap='viridis', aspect='auto')
                ax.set_title(f"Self-Attention: Layer {i}, Head {h}")
                ax.set_xlabel("Key: Next State Tokens")
                ax.set_ylabel("Query: Next State Tokens")
                fig.colorbar(im, ax=ax)
                plt.tight_layout()
                plot_filename_self = Path(output_dir) / f"self_attention_layer_{i}_head_{h}.png"
                plt.savefig(plot_filename_self)
                plt.close(fig)

    print(f"\nAll attention maps have been saved to the '{output_dir}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention maps from a trained Transformer World Model.")
    parser.add_argument(
        "--model-path", type=str, default=WM_CHECKPOINT_FILENAME_TRANSFORMER,
        help=f"Path to the trained world model checkpoint. Default: {WM_CHECKPOINT_FILENAME_TRANSFORMER}"
    )
    parser.add_argument(
        "--vq-vae-path", type=str, default=VQ_VAE_CHECKPOINT_FILENAME,
        help=f"Path to the VQ-VAE checkpoint. Default: {VQ_VAE_CHECKPOINT_FILENAME}"
    )
    parser.add_argument(
        "--ppo-path", type=str, default=SB3_MODEL_PATH,
        help=f"Path to the PPO agent checkpoint. Default: {SB3_MODEL_PATH}"
    )
    parser.add_argument(
        "--output-dir", type=str, default=IMAGES_DIR / "attention_maps",
        help="Directory to save the output plots."
    )
    args = parser.parse_args()

    draw_attention_maps(args.model_path, args.vq_vae_path, args.ppo_path, args.output_dir)
