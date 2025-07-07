import os
import pygame
import torch
import numpy as np
import cv2

from utils import (
    WM_CHECKPOINT_FILENAME_GRU, VQ_VAE_CHECKPOINT_FILENAME, ACTION_DIM
)
from vq_conv_vae import VQVAE, EMBEDDING_DIM, NUM_EMBEDDINGS
from world_model import WorldModelGRU, GRU_HIDDEN_DIM
from dreaming_render import get_starting_state_from_sequence


# --- Function to draw key presses ---
def draw_key_presses(screen, keys, screen_width, screen_height):
    """
    Draws indicators for key presses on the screen.

    Args:
        screen: The pygame.Surface to draw on.
        keys: A dictionary with the state of each key ("up", "down", "left", "right").
        screen_width: The width of the screen.
        screen_height: The height of the screen.
    """
    # Colors for the indicators
    KEY_ON_COLOR = (50, 205, 50)  # Lime Green for pressed
    KEY_OFF_COLOR = (105, 105, 105)  # Dim Gray for not pressed
    KEY_BG_COLOR = (30, 30, 30)  # Background for the D-pad

    # D-pad geometry
    key_size = 40
    key_spacing = 25
    d_pad_center_x = screen_width // 2
    d_pad_center_y = screen_height - 80  # Position it near the bottom-center

    # Define rectangle positions for a D-pad layout
    up_rect = pygame.Rect(d_pad_center_x - key_size // 2, d_pad_center_y - key_size - 5, key_size, key_size)
    down_rect = pygame.Rect(d_pad_center_x - key_size // 2, d_pad_center_y, key_size, key_size)
    left_rect = pygame.Rect(d_pad_center_x - key_size - key_spacing, d_pad_center_y, key_size, key_size)
    right_rect = pygame.Rect(d_pad_center_x + key_spacing, d_pad_center_y, key_size, key_size)

    # Draw background for better visibility
    bg_rect = pygame.Rect(left_rect.left - 5, up_rect.top - 5, key_size * 3 + key_spacing,
                          key_size * 2 + key_spacing)
    pygame.draw.rect(screen, KEY_BG_COLOR, bg_rect, border_radius=10)

    # Draw each key indicator based on its state
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["up"] else KEY_OFF_COLOR, up_rect, border_radius=5)
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["down"] else KEY_OFF_COLOR, down_rect, border_radius=5)
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["left"] else KEY_OFF_COLOR, left_rect, border_radius=5)
    pygame.draw.rect(screen, KEY_ON_COLOR if keys["right"] else KEY_OFF_COLOR, right_rect, border_radius=5)


def play_dream():
    """
    Main function to run the interactive dream environment.
    """
    # --- Configuration ---
    SCREEN_WIDTH = 1024
    SCREEN_HEIGHT = 1024
    FPS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Models ---
    print("Loading models...")
    # World Model
    world_model = WorldModelGRU(
        latent_dim=EMBEDDING_DIM,
        codebook_size=NUM_EMBEDDINGS,
        action_dim=ACTION_DIM,
        hidden_dim=GRU_HIDDEN_DIM
    ).to(DEVICE)
    world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))
    world_model.eval()

    # VQ-VAE Model
    vq_vae = VQVAE(embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS).to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
    vq_vae.eval()

    # --- Prime the Initial State ---
    image_files = sorted([os.path.join("./data/init_frames", f) for f in os.listdir("./data/init_frames")])
    priming_sequence = image_files[:10]
    hidden_state, _, initial_frame_tensor = get_starting_state_from_sequence(priming_sequence, world_model, vq_vae,
                                                                             DEVICE)

    # The first frame for display, as a NumPy array [H, W, C]
    current_frame_np = initial_frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # --- Pygame Initialization ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Playing in the Dream")
    clock = pygame.time.Clock()

    # --- Game Loop Variables ---
    running = True
    keys_pressed = {
        "up": False, "down": False, "left": False, "right": False
    }

    print("\n--- Starting Interactive Dream ---")
    print("Controls: Arrow Keys to drive. ESC or close window to quit.")

    while running:
        # --- Handle User Input (Pygame Events) ---
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

        # --- Create Action Tensor from Keyboard State ---
        steer, gas, brake = 0.0, 0.0, 0.0
        if keys_pressed["up"]: gas = 1.0
        if keys_pressed["down"]: brake = 0.8  # Brakes are usually strong
        if keys_pressed["left"]: steer = -1.0
        if keys_pressed["right"]: steer = 1.0

        action_np = np.array([steer, gas, brake], dtype=np.float32)
        action_tensor = torch.tensor(action_np, device=DEVICE).unsqueeze(0)

        # --- World Model Step ---
        with torch.no_grad():
            # Get next frame prediction and reward from the world model
            pred_logits, pred_reward, _, next_hidden_state = world_model(action_tensor, hidden_state)

            # Decode the predicted latents into an image
            b, h, w, c = pred_logits.shape
            logits_flat = pred_logits.reshape(b, h * w, c)
            predicted_indices = torch.distributions.Categorical(logits=logits_flat).sample()
            quantized_vectors = vq_vae.vq_layer.embedding(predicted_indices)
            quantized_grid = quantized_vectors.reshape(b, h, w, -1).permute(0, 3, 1, 2)
            decoded_image = vq_vae.decoder(quantized_grid)

            # Update the current frame and hidden state
            current_frame_np = (decoded_image.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255).to(
                torch.uint8).cpu().numpy()
            hidden_state = next_hidden_state

        # --- Prepare Frame for Display ---
        # Upscale the frame
        frame_large_gray = cv2.resize(current_frame_np, (SCREEN_WIDTH, SCREEN_HEIGHT),
                                      interpolation=cv2.INTER_NEAREST)

        # Convert to RGB for Pygame
        frame_large_rgb = cv2.cvtColor(frame_large_gray, cv2.COLOR_GRAY2RGB)

        # Draw the predicted reward text on the RGB frame
        reward_text = f"Reward: {pred_reward.item():.2f}"
        cv2.putText(frame_large_rgb, reward_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # --- Render to Screen ---
        # Pygame uses a different orientation, so we need to rotate and flip
        surface = pygame.surfarray.make_surface(frame_large_rgb.transpose(1, 0, 2))
        screen.blit(surface, (0, 0))

        # --- Draw the on-screen key display ---
        draw_key_presses(screen, keys_pressed, SCREEN_WIDTH, SCREEN_HEIGHT)

        pygame.display.flip()

        # --- Control Frame Rate ---
        clock.tick(FPS)

    # --- Cleanup ---
    pygame.quit()
    print("Dream finished.")


if __name__ == "__main__":
    play_dream()
