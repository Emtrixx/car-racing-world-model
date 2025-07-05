import os
from collections import deque
from typing import List

import imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from play_game_sb3 import SB3_MODEL_PATH
from utils import WM_CHECKPOINT_FILENAME_GRU, VQ_VAE_CHECKPOINT_FILENAME, ACTION_DIM
from vq_conv_vae import VQVAE, EMBEDDING_DIM, NUM_EMBEDDINGS
from world_model import WorldModelGRU, GRU_HIDDEN_DIM


def get_starting_state_from_image(image_path: str, world_model: WorldModelGRU, vq_vae: VQVAE, device):
    """
    Loads an image, encodes it, and uses it to prime the world model's hidden state.

    Args:
        image_path (str): Path to the pre-processed sample image.
        world_model (WorldModelGRU): The trained world model.
        vq_vae (VQVAE): The trained VQ-VAE.
        device: The torch device.

    Returns:
        tuple: A tuple containing the primed hidden state and the first frame tensor.
    """
    print(f"Initializing dream from image: {image_path}")
    # Load and process the image
    frame_np = imageio.imread(image_path)  # Reads as (H, W) or (H, W, C)
    frame_tensor = torch.tensor(frame_np, dtype=torch.float32, device=device)

    # Normalize if your VQ-VAE expects it (e.g., to 0-1 range)
    frame_tensor = frame_tensor / 255.0

    # Ensure correct shape (B, C, H, W)
    if len(frame_tensor.shape) == 2:  # Grayscale (H, W)
        frame_tensor = frame_tensor.unsqueeze(0)  # Add channel dim -> (1, H, W)
    frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dim -> (1, 1, H, W)

    # Get the ground truth tokens for this image
    with torch.no_grad():
        first_frame_tensor, _, _, indices = vq_vae(frame_tensor)
        indices = indices.view(1, -1)  # Flatten to [1, 16]

    # Prime the hidden state by running the model with teacher forcing
    # This generates a hidden state that is consistent with having seen the image.
    zero_hidden_state = world_model.get_initial_hidden_state(batch_size=1, device=device)
    dummy_action = torch.zeros(1, world_model.action_embedding.in_features, device=device)

    with torch.no_grad():
        _, _, _, primed_hidden_state = world_model(
            dummy_action, zero_hidden_state, ground_truth_tokens=indices
        )

    return primed_hidden_state, first_frame_tensor


def get_starting_state_from_sequence(image_paths: List[str], world_model: WorldModelGRU, vq_vae: VQVAE, device):
    """
    Loads a sequence of images, encodes them, and processes them sequentially
    to prime the world model's hidden state with a rich context.

    Args:
        image_paths (List[str]): A list of paths to the pre-processed sample images, in order.
        world_model (WorldModelGRU): The trained world model.
        vq_vae (VQVAE): The trained VQ-VAE.
        device: The torch device.

    Returns:
        tuple: A tuple containing the final primed hidden state and the reconstructed
               tensor of the *last* image in the sequence.
    """
    print(f"Initializing dream from a sequence of {len(image_paths)} images...")

    # Initialize the hidden state for the GRU
    hidden_state = world_model.get_initial_hidden_state(batch_size=1, device=device)
    last_frame_reconstruction = None

    with torch.no_grad():
        # 2. Loop through each image in the sequence
        for image_path in image_paths:
            # Load and process the image
            frame_np = imageio.imread(image_path)
            frame_tensor = torch.tensor(frame_np, dtype=torch.float32, device=device) / 255.0

            # Ensure correct shape (B, C, H, W)
            if len(frame_tensor.shape) == 2:
                frame_tensor = frame_tensor.unsqueeze(0)  # Add channel dim
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dim

            # Get the ground truth tokens for the current image
            reconstruction, _, _, indices = vq_vae(frame_tensor)
            indices = indices.view(1, -1)  # Flatten to [1, 16]

            # Store the latest reconstruction to return it later
            last_frame_reconstruction = reconstruction

            # 3. Update the hidden state using the current image's tokens
            # We provide the current hidden state and the tokens from the new frame.
            # The model returns the *next* hidden state, which we use in the next loop iteration.
            dummy_action = torch.zeros(1, world_model.action_embedding.in_features, device=device)
            _, _, _, hidden_state = world_model(
                dummy_action, hidden_state, ground_truth_tokens=indices
            )

    print("Priming complete. Final hidden state captured.")
    # Return the final hidden state and the last frame
    return hidden_state, last_frame_reconstruction, frame_tensor


def dream(world_model,
          vq_vae: VQVAE,
          ppo_agent,
          initial_hidden_state,
          initial_frame: np.ndarray,
          num_steps: int,
          num_stack: int = 4,
          device=torch.device("cpu"),
          ):
    """
    Generates a sequence of imagined frames by running the world model in a loop.

    Args:
        world_model: The trained GRU world model.
        vq_vae: The trained VQ-VAE for decoding frames.
        ppo_agent: The trained PPO agent (e.g., from Stable Baselines 3).
        initial_hidden_state: The starting hidden state for the world model.
        initial_frame (torch.Tensor): The first single frame (H, W) to seed the dream.
        num_steps (int): The number of steps to dream for.
        num_stack (int): The number of frames to stack for the agent's observation.
        device: The torch device to run the models on.

    Returns:
        list: A list of generated frames as NumPy arrays (H, W, C).
    """
    print(f"Dreaming for {num_steps} steps...")
    world_model.eval()
    vq_vae.eval()

    # Store the generated frames here
    dreamed_frames = []

    hidden_state = initial_hidden_state.to(device)

    # frame stacking for the agent
    frame_buffer = deque([initial_frame] * num_stack, maxlen=num_stack)

    with torch.no_grad():
        for step in range(num_steps):
            if step % 50 == 0:
                print(f"  Dream step {step}/{num_steps}")

            agent_obs = np.array(frame_buffer)
            action, _ = ppo_agent.predict(agent_obs, deterministic=True)
            action_tensor = torch.tensor(action, device=device).float().unsqueeze(0)

            # Run the world model for one step in inference mode
            # We don't provide ground_truth_tokens, so it uses its own predictions.
            pred_logits, _, _, next_hidden_state = world_model(action_tensor, hidden_state, ground_truth_tokens=None)

            # Get the predicted next state tokens by sampling from the logits
            # Reshape logits to [batch_size, num_tokens, codebook_size] for sampling
            b, h, w, c = pred_logits.shape
            logits_flat = pred_logits.reshape(b, h * w, c)

            # Sample from the distribution to get the indices of the next tokens
            predicted_indices = torch.distributions.Categorical(logits=logits_flat).sample()  # Shape: [1, 16]

            # Decode the predicted tokens back into an image
            # First, get the quantized vectors from the indices using the VQ-VAE's codebook
            quantized_vectors = vq_vae.vq_layer.embedding(predicted_indices)  # Shape: [1, 16, latent_dim]

            # Reshape to the grid format the decoder expects
            quantized_grid = quantized_vectors.reshape(b, h, w, -1)  # Shape: [1, 4, 4, latent_dim]
            # The VQ-VAE decoder expects BCHW,so permute dimensions
            quantized_grid_permuted = quantized_grid.permute(0, 3, 1, 2)  # Shape: [1, latent_dim, 4, 4]

            # Decode to get the image
            decoded_image = vq_vae.decoder(quantized_grid_permuted)  # Shape: [1, 3, 96, 96]
            frame_buffer.append(decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy())

            # Post-process the frame for saving
            # Remove batch dimension, convert to HWC, and scale to 0-255 uint8
            frame = decoded_image.squeeze(0).permute(1, 2, 0)
            frame = (frame * 255).clamp(0, 255).to(torch.uint8)
            dreamed_frames.append(frame.cpu().numpy())

            # Prepare for the next step
            hidden_state = next_hidden_state

    print("Dreaming complete.")
    return dreamed_frames


def generate_video(frames, output_path, fps=30):
    """
    Takes a list of frames and saves them as a video file.

    Args:
        frames (list): A list of NumPy array frames (H, W, C).
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


# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    DREAM_STEPS = 300
    SAMPLE_IMAGE_DIR = "./assets/sample_images/"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = "videos"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load world model
    print("Loading trained models...")
    world_model = WorldModelGRU(
        latent_dim=EMBEDDING_DIM,
        codebook_size=NUM_EMBEDDINGS,
        action_dim=ACTION_DIM,
        hidden_dim=GRU_HIDDEN_DIM
    ).to(DEVICE)
    world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))

    # Load VQ-VAE model
    vq_vae = VQVAE(embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS).to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))

    # --- Load Trained SB3 PPO Agent ---
    print(f"Loading trained SB3 PPO agent from: {SB3_MODEL_PATH}")
    if not SB3_MODEL_PATH.exists():
        print(f"ERROR: SB3 PPO Model not found at {SB3_MODEL_PATH}")
        exit(1)
    try:
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE)
        print(f"Successfully loaded SB3 PPO agent. Agent device: {ppo_agent.device}")
    except Exception as e:
        print(f"ERROR loading SB3 PPO agent: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # --- setup for dreaming from a single image ---
    # Select a random image and prime the starting state
    # image_files = [f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.endswith('.png') and not f.startswith('frame_')]
    # if not image_files:
    #     raise FileNotFoundError(f"No sample images found in {SAMPLE_IMAGE_DIR}. Please add some.")
    #
    # # random_image_file = random.choice(image_files)
    # random_image_file = "frame_0022.png"
    # start_image_path = os.path.join(SAMPLE_IMAGE_DIR, random_image_file)
    #
    # primed_h, first_frame_tensor = get_starting_state_from_image(start_image_path, world_model, vq_vae, DEVICE)

    # --- setup for dreaming from a sequence of images ---
    image_files = sorted([os.path.join("./data/init_frames", f) for f in os.listdir("./data/init_frames")])
    priming_sequence = image_files[:10]
    primed_h, last_frame_reconstruction, frame_tensor = get_starting_state_from_sequence(priming_sequence, world_model,
                                                                                         vq_vae, DEVICE)

    initial_a = torch.zeros(1, ACTION_DIM, device=DEVICE)

    # Generate the dream sequence
    initial_frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    dreamed_frames_list = dream(
        world_model=world_model,
        vq_vae=vq_vae,
        ppo_agent=ppo_agent,
        initial_frame=initial_frame_np,
        num_steps=DREAM_STEPS,
        initial_hidden_state=primed_h,
        device=DEVICE
    )

    # Save the frames as a video
    if dreamed_frames_list:
        video_path = os.path.join(OUTPUT_DIR, "world_model_dream.mp4")
        generate_video(dreamed_frames_list, video_path, fps=7)
