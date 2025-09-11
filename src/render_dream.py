import argparse
import random
import os
from collections import deque
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm

from src.dream_env import GruDreamEnv
from src.dream_env_transformer import DreamEnvTransformer
from src.train_dyna_loop import get_vq_indices
from src.transformer_world_model import WorldModelTransformer
from src.utils import make_env_sb3, NUM_STACK, DEVICE, VQ_VAE_CHECKPOINT_FILENAME, ENV_NAME, FrameStackWrapper, \
    VIDEO_DIR, ACTION_DIM, WM_CHECKPOINT_FILENAME_GRU, WM_CHECKPOINT_FILENAME_TRANSFORMER
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM
from src.world_model import WorldModelGRU
from src.play_game_sb3 import SB3_MODEL_PATH
from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS
from src.train_transformer_world_model import HISTORY_LENGTH


def generate_video(frames, output_path, fps=8):
    """
    Takes a list of frames and saves them as a video file using imageio.

    Args:
        frames (list): A list of NumPy array frames (H, W, C) in RGB format.
        output_path (str): The path to save the video file (e.g., 'dream.mp4').
        fps (int): Frames per second for the output video.
    """
    print(f"Generating video with {len(frames)} frames...")
    try:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Video saved successfully to {output_path}")
    except ImportError:
        print("Error: `imageio` and `imageio-ffmpeg` are required.")
        print("Please install them with: pip install imageio[ffmpeg]")
    except Exception as e:
        print(f"An error occurred during video generation: {e}")


def collect_real_buffer(env, vq_vae, ppo_agent, buffer_size, device):
    """Collects a small buffer of real experience to initialize the dream environment."""
    print(f"Collecting a buffer of {buffer_size} real transitions to initialize dream...")
    real_buffer = deque(maxlen=buffer_size)
    obs, _ = env.reset()
    is_first_step = True

    for _ in tqdm(range(buffer_size), desc="Collecting real experience"):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dream sequences from a trained world model.")
    parser.add_argument('--model_type', type=str, required=True, choices=['gru', 'transformer'],
                        help="Type of world model to use ('gru' or 'transformer').")
    parser.add_argument('--dream_steps', type=int, default=500,
                        help="Total number of steps to dream for.")
    parser.add_argument('--horizon', type=int, default=100,
                        help="Maximum steps per dream episode before truncation.")
    parser.add_argument('--history_len', type=int, default=HISTORY_LENGTH,
                        help="History length for the Transformer model.")
    parser.add_argument('--real_buffer_size', type=int, default=500,
                        help="Size of the real experience buffer for initialization.")
    parser.add_argument('--output_filename', type=str, default=None,
                        help="Custom filename for the output video.")
    parser.add_argument('--deterministic', action='store_true',
                        help="Run PPO agent in deterministic mode.")
    args = parser.parse_args()

    print(f"--- Configuration ---")
    print(f"Model Type: {args.model_type}")
    print(f"Dream Steps: {args.dream_steps}")
    print(f"Deterministic PPO: {args.deterministic}")
    print(f"---------------------")

    # --- Constants and Setup ---
    DISPLAY_SIZE = 512
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # --- Load Common Models (VQ-VAE, PPO Agent) ---
    print("Loading common models (VQ-VAE and PPO agent)...")
    real_env = make_env_sb3(env_id=ENV_NAME, frame_stack_num=NUM_STACK)

    vq_vae = VQVAE().to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
    vq_vae.eval()

    ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=real_env)

    # --- Collect Real Data for Initialization ---
    real_experience_buffer = collect_real_buffer(real_env, vq_vae, ppo_agent, args.real_buffer_size, DEVICE)
    real_env.close()

    # --- Load World Model and Create Dream Environment ---
    print(f"Loading {args.model_type.upper()} world model and creating dream environment...")
    dream_env = None
    default_video_filename = ""

    if args.model_type == 'gru':
        world_model = WorldModelGRU(
            latent_dim=VQVAE_EMBEDDING_DIM,
            codebook_size=VQVAE_NUM_EMBEDDINGS,
            action_dim=ACTION_DIM,
        ).to(DEVICE)
        world_model = torch.compile(world_model)
        world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))
        world_model.eval()

        dream_env = GruDreamEnv(
            world_model=world_model,
            vq_vae=vq_vae,
            device=DEVICE,
            seed=random.randint(0, 2 ** 31 - 1),
            horizon=args.horizon,
            real_buffer=real_experience_buffer
        )
        default_video_filename = "world_model_dream_gru.mp4"

    elif args.model_type == 'transformer':
        world_model = WorldModelTransformer(
            vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
            action_dim=ACTION_DIM,
        ).to(DEVICE)
        world_model = torch.compile(world_model)
        if Path(WM_CHECKPOINT_FILENAME_TRANSFORMER).exists():
            world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_TRANSFORMER, map_location=DEVICE))
        else:
            raise FileNotFoundError(
                f"Transformer world model checkpoint not found: {WM_CHECKPOINT_FILENAME_TRANSFORMER}")
        world_model.eval()

        dream_env = DreamEnvTransformer(
            world_model=world_model,
            vq_vae=vq_vae,
            device=DEVICE,
            seed=random.randint(0, 2 ** 31 - 1),
            history_length=args.history_len,
            horizon=args.horizon,
            real_buffer=real_experience_buffer
        )
        default_video_filename = "world_model_dream_transformer.mp4"

    dream_env = FrameStackWrapper(dream_env, num_stack=NUM_STACK)

    # --- Run Dream Sequence ---
    print(f"Dreaming for {args.dream_steps} steps...")
    obs, _ = dream_env.reset()
    dreamed_frames_for_video = []

    for step in tqdm(range(args.dream_steps), desc="Dreaming"):
        action, _ = ppo_agent.predict(obs, deterministic=args.deterministic)
        obs, reward, done, truncated, info = dream_env.step(action)

        # --- Visualization ---
        frame_np = (obs[-1] * 255).clip(0, 255).astype(np.uint8)
        frame_large = cv2.resize(frame_np, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.putText(frame_large, f"Reward: {reward:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        dreamed_frames_for_video.append(frame_large)

        if done or truncated:
            obs, _ = dream_env.reset()

    print("Dreaming complete.")
    dream_env.close()

    # --- Save video ---
    if dreamed_frames_for_video:
        video_path = VIDEO_DIR / (args.output_filename if args.output_filename else default_video_filename)
        generate_video(dreamed_frames_for_video, video_path, fps=7)
