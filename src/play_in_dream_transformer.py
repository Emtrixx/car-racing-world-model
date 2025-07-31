import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from play_game_sb3 import SB3_MODEL_PATH
from src.transformer_world_model import WorldModelTransformer
from src.utils import (
    WM_CHECKPOINT_FILENAME_TRANSFORMER, VQ_VAE_CHECKPOINT_FILENAME, ACTION_DIM, make_env_sb3, ENV_NAME, NUM_STACK
)
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM
from utils import preprocess_observation, DATA_DIR


# --- Function to draw key presses ---
def draw_key_presses(screen, keys, screen_width, screen_height):
    """
    Draws indicators for key presses on the screen.
    """
    KEY_ON_COLOR = (50, 205, 50)
    KEY_OFF_COLOR = (105, 105, 105)
    KEY_BG_COLOR = (30, 30, 30)
    key_size = 40
    key_spacing = 25
    d_pad_center_x = screen_width // 2
    d_pad_center_y = screen_height - 80
    up_rect = pygame.Rect(d_pad_center_x - key_size // 2, d_pad_center_y - key_size - 5, key_size, key_size)
    down_rect = pygame.Rect(d_pad_center_x - key_size // 2, d_pad_center_y, key_size, key_size)
    left_rect = pygame.Rect(d_pad_center_x - key_size - key_spacing, d_pad_center_y, key_size, key_size)
    right_rect = pygame.Rect(d_pad_center_x + key_spacing, d_pad_center_y, key_size, key_size)
    bg_rect = pygame.Rect(left_rect.left - 5, up_rect.top - 5, key_size * 3 + key_spacing, key_size * 2 + key_spacing)
    pygame.draw.rect(screen, KEY_BG_COLOR, bg_rect, border_radius=10)
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["up"] else KEY_OFF_COLOR, up_rect, border_radius=5)
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["down"] else KEY_OFF_COLOR, down_rect, border_radius=5)
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["left"] else KEY_OFF_COLOR, left_rect, border_radius=5)
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["right"] else KEY_OFF_COLOR, right_rect, border_radius=5)


def get_initial_tokens_from_image(image_path, vq_vae, device):
    """Loads an image, preprocesses it, and gets the initial VQ-VAE tokens."""
    frame = cv2.imread(image_path)
    # preprocess_frame expects RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # processed_frame is a normalized grayscale numpy array (H, W, 1)
    processed_frame = preprocess_observation(frame_rgb)

    frame_tensor = torch.tensor(processed_frame, dtype=torch.float32, device=device)
    # from (H, W, 1) to (1, 1, H, W)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        z_continuous = vq_vae.encoder(frame_tensor)
        z_continuous = vq_vae._pre_vq_conv(z_continuous)
        _, _, _, initial_tokens = vq_vae.vq_layer(z_continuous)

    # Return tokens and the initial frame for display
    return initial_tokens.view(-1), processed_frame


def play_dream_transformer(autoplay=False, deterministic=False):
    """
    Main function to run the interactive dream environment with the Transformer World Model.
    """

    # --- Configuration ---
    SCREEN_WIDTH = 1024
    SCREEN_HEIGHT = 1024
    FPS = 8  # Transformer can be a bit slower
    HISTORY_LEN = 32  # Must match the history length the model was trained with
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Models ---
    print("Loading models...")
    # VQ-VAE Model
    vq_vae = VQVAE().to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
    vq_vae.eval()

    # PPO Agent (optional, if you want to use it for actions)
    env = make_env_sb3(env_id=ENV_NAME, frame_stack_num=NUM_STACK)
    ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=env)

    # Transformer World Model
    world_model = WorldModelTransformer(
        vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
    ).to(DEVICE)
    world_model = torch.compile(world_model)  # Compile for performance
    if Path(WM_CHECKPOINT_FILENAME_TRANSFORMER).exists():
        world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_TRANSFORMER, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"World model checkpoint not found: {WM_CHECKPOINT_FILENAME_TRANSFORMER}")
    world_model.eval()

    # --- Prime the Initial State ---
    init_image_path = DATA_DIR / "init_frames/frame_0001.png"  # Use the first frame
    if not os.path.exists(init_image_path):
        raise FileNotFoundError(f"Initial image not found at {init_image_path}. Please generate initial frames.")

    initial_tokens, current_frame_np = get_initial_tokens_from_image(init_image_path, vq_vae, DEVICE)

    # --- Initialize History ---
    action_history = deque(maxlen=HISTORY_LEN)
    token_history = deque(maxlen=HISTORY_LEN)

    zero_action = torch.zeros(ACTION_DIM, device=DEVICE)

    # Pre-fill the history deques
    for _ in range(HISTORY_LEN):
        action_history.append(zero_action)
        token_history.append(initial_tokens)

    agent_frame_buffer = deque(current_frame_np, maxlen=NUM_STACK)
    for _ in range(NUM_STACK):
        agent_frame_buffer.append(current_frame_np)
    # --- Pygame Initialization ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Playing in the Transformer Dream")
    clock = pygame.time.Clock()

    # --- Game Loop Variables ---
    running = True
    keys_pressed = {"up": False, "down": False, "left": False, "right": False}
    current_reward = 0.0

    print("\n--- Starting Interactive Transformer Dream ---")
    print("Controls: Arrow Keys to drive. ESC or close window to quit.")

    while running:
        # --- Handle User Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: keys_pressed["up"] = True
                if event.key == pygame.K_DOWN: keys_pressed["down"] = True
                if event.key == pygame.K_LEFT: keys_pressed["left"] = True
                if event.key == pygame.K_RIGHT: keys_pressed["right"] = True
                if event.key == pygame.K_ESCAPE: running = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_pressed["up"] = False
                if event.key == pygame.K_DOWN: keys_pressed["down"] = False
                if event.key == pygame.K_LEFT: keys_pressed["left"] = False
                if event.key == pygame.K_RIGHT: keys_pressed["right"] = False

        # --- Create Action Tensor ---
        steer, gas, brake = 0.0, -1.0, -1.0
        if keys_pressed["up"]: gas = 0.8
        if keys_pressed["down"]: brake = 0.2
        if keys_pressed["left"]: steer = -1.0
        if keys_pressed["right"]: steer = 1.0
        action_np = np.array([steer, gas, brake], dtype=np.float32)
        current_action_tensor = torch.tensor(action_np, device=DEVICE)

        if autoplay and not keys_pressed["up"] and not keys_pressed["down"] and not keys_pressed["left"] and not \
                keys_pressed[
                    "right"]:
            stacked_frames = np.stack([np.array(frame) for frame in agent_frame_buffer], axis=0)
            current_action, _ = ppo_agent.predict(stacked_frames, deterministic=False)
            current_action_tensor = torch.tensor(current_action, device=DEVICE).float()

        # Print the current action for debugging
        # print(f"Gas: {current_action_tensor[1]:.2f}, "
        #       f"Brake: {current_action_tensor[2]:.2f}, "
        #       f"Steer: {current_action_tensor[0]:.2f}")

        # Update action history
        action_history.append(current_action_tensor)

        # --- World Model Step ---
        with torch.no_grad():
            action_history_tensor = torch.stack(list(action_history)).unsqueeze(0)
            token_history_tensor = torch.stack(list(token_history)).unsqueeze(0)

            pred_logits, pred_reward, pred_done_logits, generated_tokens = world_model(
                action_history_tensor, token_history_tensor
            )

            b, h, w, c = pred_logits.shape  # Get grid shape from logits
            # Decode the generated tokens into an image
            if deterministic:
                tokens_for_decoding = generated_tokens.squeeze(0)
            else:
                # Reshape for sampling: from [b, h, w, num_embeddings] to [b*h*w, num_embeddings]
                pred_probs = torch.softmax(pred_logits.view(-1, c), dim=-1)
                # Sample one token for each position
                tokens_for_decoding = torch.multinomial(pred_probs, num_samples=1).squeeze(1)

            quantized_vectors = vq_vae.vq_layer.embeddings[tokens_for_decoding]
            quantized_grid = quantized_vectors.view(h, w, -1)
            quantized_grid_permuted = quantized_grid.permute(2, 0, 1).unsqueeze(0)
            decoded_image = vq_vae.decoder(quantized_grid_permuted)
            agent_frame_buffer.append(decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy())

            # Update state for the next loop
            token_history.append(generated_tokens.squeeze(0))
            current_frame_np = (decoded_image.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255).to(
                torch.uint8).cpu().numpy()
            current_reward = pred_reward.item()

        # --- Prepare Frame for Display ---
        # Upscale and convert to RGB for display
        if current_frame_np.shape[2] == 1:
            frame_large_gray = cv2.resize(current_frame_np, (SCREEN_WIDTH, SCREEN_HEIGHT),
                                          interpolation=cv2.INTER_NEAREST)
            frame_large_rgb = cv2.cvtColor(frame_large_gray, cv2.COLOR_GRAY2RGB)
        else:
            frame_large_rgb = cv2.resize(current_frame_np, (SCREEN_WIDTH, SCREEN_HEIGHT),
                                         interpolation=cv2.INTER_NEAREST)

        # Draw the predicted reward text on the RGB frame
        reward_text = f"Reward: {current_reward:.3f}"
        cv2.putText(frame_large_rgb, reward_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # --- Render to Screen ---
        surface = pygame.surfarray.make_surface(frame_large_rgb.transpose(1, 0, 2))
        screen.blit(surface, (0, 0))
        draw_key_presses(screen, keys_pressed, SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.display.flip()

        # --- Control Frame Rate ---
        clock.tick(FPS)

    # --- Cleanup ---
    pygame.quit()
    print("Transformer dream finished.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Play in the Transformer Dream")
    arg_parser.add_argument("--autoplay", action="store_true", help="Enable autoplay mode")
    arg_parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (no randomness)"
                            )
    args = arg_parser.parse_args()

    play_dream_transformer(autoplay=args.autoplay, deterministic=args.deterministic)
