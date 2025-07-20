import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO

from play_game_sb3 import SB3_MODEL_PATH
from src.transformer_world_model import WorldModelTransformer
from src.utils import make_env_sb3, NUM_STACK, DEVICE, VQ_VAE_CHECKPOINT_FILENAME, ENV_NAME
from src.vq_conv_vae import VQVAE
from utils import WM_CHECKPOINT_FILENAME_TRANSFORMER, ACTION_DIM, VIDEO_DIR
from vq_conv_vae import VQVAE_EMBEDDING_DIM


def get_initial_obs_and_tokens(env, vq_vae, device):
    """Resets the environment and gets the initial observation and corresponding tokens."""
    obs, _ = env.reset()
    latest_frame = obs[-1]
    frame_tensor = torch.tensor(latest_frame, dtype=torch.float32, device=device)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        z_continuous = vq_vae.encoder(frame_tensor)
        z_continuous = vq_vae._pre_vq_conv(z_continuous)
        _, _, _, initial_tokens = vq_vae.vq_layer(z_continuous)

    # Return tokens as a 1D tensor [num_tokens]
    return obs, initial_tokens.view(-1)


def dream_with_history(
        world_model: WorldModelTransformer,
        vq_vae: VQVAE,
        ppo_agent,
        initial_obs: np.ndarray,
        initial_tokens: torch.Tensor,
        history_length: int,
        num_steps: int,
        num_stack: int = 4,
        device=torch.device("cpu"),
):
    """
    Generates a sequence of imagined frames using the history-aware WorldModelTransformer.
    """
    DISPLAY_SIZE = 512
    print(f"Dreaming for {num_steps} steps with a history length of {history_length}...")
    world_model.eval()
    vq_vae.eval()

    dreamed_frames_for_video = []

    # --- Initialize History ---
    action_history = deque(maxlen=history_length)
    token_history = deque(maxlen=history_length)

    # Initialize with placeholder data
    zero_action = torch.zeros(ppo_agent.action_space.shape, device=device)
    zero_tokens = torch.zeros_like(initial_tokens, device=device)  # Shape: [num_tokens]

    # Pre-fill the history deques completely with zeros
    for _ in range(history_length):
        action_history.append(zero_action)
        token_history.append(zero_tokens)

    # Now, replace the last element with the actual initial state
    token_history[-1] = initial_tokens.to(device)

    # The agent's observation buffer starts with the real observation
    agent_frame_buffer = deque(initial_obs, maxlen=num_stack)

    with torch.no_grad():
        for step in range(num_steps):
            if step % 25 == 0:
                print(f"  Dream step {step}/{num_steps}")

            # --- Get Action ---
            # The action for the *current* step is based on the *current* agent buffer
            current_action, _ = ppo_agent.predict(np.array(agent_frame_buffer), deterministic=True)
            current_action_tensor = torch.tensor(current_action, device=device).float()

            # Add the current action to its history. This action corresponds to the last token in token_history
            action_history.append(current_action_tensor)

            # --- Prepare Tensors from History ---
            # The history is now always full, ensuring consistent tensor shapes.
            action_history_tensor = torch.stack(list(action_history)).unsqueeze(0)  # [1, H, action_dim]
            token_history_tensor = torch.stack(list(token_history)).unsqueeze(0)  # [1, H, num_tokens]

            # --- Predict the Next State with the World Model ---
            pred_logits, pred_reward, pred_done_logits, generated_tokens = world_model(
                action_history_tensor, token_history_tensor
            )

            # --- Decode to a Frame ---
            tokens_for_decoding = generated_tokens.squeeze(0)  # [num_tokens]
            b, h, w, c = pred_logits.shape
            quantized_vectors = vq_vae.vq_layer.embeddings[tokens_for_decoding]
            quantized_grid = quantized_vectors.view(h, w, -1)
            quantized_grid_permuted = quantized_grid.permute(2, 0, 1).unsqueeze(0)
            decoded_image = vq_vae.decoder(quantized_grid_permuted)

            # --- Visualization ---
            frame_for_viz = decoded_image.squeeze(0).permute(1, 2, 0)
            frame_np = (frame_for_viz * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            frame_large = cv2.resize(frame_np, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
            reward_val = pred_reward.item()
            cv2.putText(frame_large, f"Reward: {reward_val:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            dreamed_frames_for_video.append(frame_large)

            # --- Update State for Next Loop ---
            # 1. Update the agent's frame buffer with the newly dreamed frame
            agent_frame_buffer.append(decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy())

            # 2. Update the world model's token history with the predicted tokens
            token_history.append(generated_tokens.squeeze(0))
            # The action history is updated at the start of the next loop

    print("Dreaming complete.")
    return dreamed_frames_for_video


if __name__ == '__main__':
    # --- Configuration ---
    DREAM_STEPS = 200
    HISTORY_LEN = 16  # Must match the history length the model was trained with

    # --- Load Models ---
    print("Loading models...")
    env = make_env_sb3(env_id=ENV_NAME, frame_stack_num=NUM_STACK)

    # Load VQ-VAE
    vq_vae = VQVAE().to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
    vq_vae.eval()

    # Load PPO Agent
    ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=env)

    # Load World Model
    # Make sure to provide the correct arguments used during training
    world_model = WorldModelTransformer(
        vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
    ).to(DEVICE)

    # Load your trained world model checkpoint
    if Path(WM_CHECKPOINT_FILENAME_TRANSFORMER).exists():
        world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_TRANSFORMER, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"World model checkpoint not found: {WM_CHECKPOINT_FILENAME_TRANSFORMER}")

    # --- Run Dream Sequence ---
    initial_obs_stack, initial_tokens_flat = get_initial_obs_and_tokens(env, vq_vae, DEVICE)

    dreamed_frames = dream_with_history(
        world_model=world_model,
        vq_vae=vq_vae,
        ppo_agent=ppo_agent,
        initial_obs=initial_obs_stack,
        initial_tokens=initial_tokens_flat,
        history_length=HISTORY_LEN,
        num_steps=DREAM_STEPS,
        num_stack=NUM_STACK,
        device=DEVICE,
    )

    # --- Save video ---
    if dreamed_frames:
        video_path = VIDEO_DIR / f"{ENV_NAME}_dream_sequence.mp4"
        height, width, layers = dreamed_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
        for frame in dreamed_frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()
        print(f"Dream sequence saved to {video_path}")

    env.close()
