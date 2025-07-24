import argparse
import os
from collections import deque
from typing import List, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
from stable_baselines3 import PPO

from src.play_game_sb3 import SB3_MODEL_PATH
from src.transformer_world_model import WorldModelTransformer
from src.utils import VIDEO_DIR, ASSETS_DIR, DATA_DIR, WM_CHECKPOINT_FILENAME_GRU, VQ_VAE_CHECKPOINT_FILENAME, \
    ACTION_DIM
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM, VQVAE_NUM_EMBEDDINGS
from src.world_model import WorldModelGRU


def get_starting_state_from_image(image_path: str, world_model: WorldModelGRU,
                                  vq_vae: VQVAE, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads an image, encodes it, and uses it to prime the world model's state.

    Args:
        image_path (str): Path to the pre-processed sample image.
        world_model WorldModelGRU: The trained world model.
        vq_vae (VQVAE): The trained VQ-VAE.
        device: The torch device.

    Returns:
        tuple: A tuple containing (primed_hidden_state, first_frame_tensor).
    """
    print(f"Initializing dream from image: {image_path}")
    frame_np = imageio.imread(image_path)
    frame_tensor = torch.tensor(frame_np, dtype=torch.float32, device=device) / 255.0

    # Adjust for grayscale or RGB images
    if len(frame_tensor.shape) == 2:  # Grayscale
        frame_tensor = frame_tensor.unsqueeze(0)  # Add channel dimension
    elif len(frame_tensor.shape) == 3 and frame_tensor.shape[-1] == 3:  # RGB
        frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert HWC to CHW
    frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        zero_hidden_state = world_model.get_initial_hidden_state(batch_size=1, device=device)
        first_frame_tensor, _, _, indices, _, _ = vq_vae(frame_tensor)
        indices = indices.view(1, -1)  # Flatten to [1, 16]
        generation_hidden_state_stack = world_model.encode_observation(observation_tokens=indices)

    return generation_hidden_state_stack, first_frame_tensor


def get_starting_state_from_sequence(image_paths: List[str],
                                     world_model: Union[WorldModelGRU, WorldModelTransformer],
                                     vq_vae: VQVAE, device) -> Tuple[
    Union[torch.Tensor, None], Union[torch.Tensor, None], torch.Tensor]:
    """
    Loads a sequence of images, encodes them, and processes them sequentially
    to prime the world model's state.

    Args:
        image_paths (List[str]): A list of paths to the pre-processed sample images, in order.
        world_model (Union[WorldModelGRU, WorldModelTransformer]): The trained world model.
        vq_vae (VQVAE): The trained VQ-VAE.
        device: The torch device.

    Returns:
        tuple: A tuple containing (final_primed_hidden_state, final_latent_tokens, last_frame_reconstruction).
               One of final_primed_hidden_state or final_latent_tokens will be None depending on model type.
    """
    print(f"Initializing dream from a sequence of {len(image_paths)} images...")

    last_frame_reconstruction = None
    last_tokens = None

    hidden_state = world_model.get_initial_hidden_state(batch_size=1, device=device)
    with torch.no_grad():
        for image_path in image_paths:
            frame_np = imageio.imread(image_path)
            frame_tensor = torch.tensor(frame_np, dtype=torch.float32, device=device) / 255.0
            # Adjust for grayscale or RGB images
            if len(frame_tensor.shape) == 2:  # Grayscale
                frame_tensor = frame_tensor.unsqueeze(0)  # Add channel dimension
            elif len(frame_tensor.shape) == 3 and frame_tensor.shape[-1] == 3:  # RGB
                frame_tensor = frame_tensor.permute(2, 0, 1)  # Convert HWC to CHW
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

            reconstruction, _, _, indices, _, _ = vq_vae(frame_tensor)
            indices = indices.view(1, -1)
            last_frame_reconstruction = reconstruction

            dummy_action = torch.zeros(1, world_model.action_embedding.in_features, device=device)
            _, _, _, hidden_state = world_model(
                dummy_action, hidden_state, ground_truth_next_tokens=indices
            )
    return hidden_state, None, last_frame_reconstruction


def dream_gru(world_model: WorldModelGRU,
              vq_vae: VQVAE,
              ppo_agent,
              initial_hidden_state: torch.Tensor,
              initial_frame: np.ndarray,
              num_steps: int,
              num_stack: int = 4,
              device=torch.device("cpu"),
              ):
    """
    Generates a sequence of imagined frames by running the GRU world model in a loop.

    Args:
        world_model: The trained GRU world model.
        vq_vae: The trained VQ-VAE for decoding frames.
        ppo_agent: The trained PPO agent (e.g., from Stable Baselines 3).
        initial_hidden_state: The starting hidden state for the world model.
        initial_frame (np.ndarray): The first single frame (H, W) to seed the dream.
        num_steps (int): The number of steps to dream for.
        num_stack (int): The number of frames to stack for the agent's observation.
        device: The torch device to run the models on.

    Returns:
        list: A list of generated frames as NumPy arrays (H, W, C).
    """
    DISPLAY_SIZE = 512

    print(f"Dreaming for {num_steps} steps with GRU World Model...")
    world_model.eval()
    vq_vae.eval()

    dreamed_frames = []
    hidden_state = initial_hidden_state.to(device)
    frame_buffer = deque([initial_frame] * num_stack, maxlen=num_stack)

    with torch.no_grad():
        for step in range(num_steps):
            if step % 50 == 0:
                print(f"  Dream step {step}/{num_steps}")

            agent_obs = np.array(frame_buffer)
            action, _ = ppo_agent.predict(agent_obs, deterministic=False)
            action_tensor = torch.tensor(action, device=device).float().unsqueeze(0)

            pred_logits, pred_reward, _, next_hidden_state = world_model(action_tensor, hidden_state,
                                                                         ground_truth_next_tokens=None)

            b, h, w, c = pred_logits.shape
            logits_flat = pred_logits.reshape(b, h * w, c)
            predicted_indices = torch.distributions.Categorical(logits=logits_flat).sample()

            quantized_vectors = vq_vae.vq_layer.embeddings[predicted_indices]
            quantized_grid = quantized_vectors.reshape(b, h, w, -1)
            quantized_grid_permuted = quantized_grid.permute(0, 3, 1, 2)

            decoded_image = vq_vae.decoder(quantized_grid_permuted)
            frame_buffer.append(decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy())

            frame = decoded_image.squeeze(0).permute(1, 2, 0)
            frame_np = (frame * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

            frame_np_large = cv2.resize(
                frame_np,
                (DISPLAY_SIZE, DISPLAY_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

            reward_value = pred_reward.item()
            reward_text = f"Reward: {reward_value:.2f}"

            cv2.putText(
                img=frame_np_large,
                text=reward_text,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            dreamed_frames.append(frame_np_large)
            hidden_state = next_hidden_state

    print("GRU Dreaming complete.")
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
    parser = argparse.ArgumentParser(description="Generate dream sequences using a world model.")
    parser.add_argument("--dream_steps", type=int, default=300,
                        help="Number of steps to dream for.")
    parser.add_argument("--output_filename", type=str,
                        help="Custom filename for the output video. Defaults based on model type.")
    args = parser.parse_args()

    DREAM_STEPS = args.dream_steps
    SAMPLE_IMAGE_DIR = ASSETS_DIR / "sample_images"
    INIT_FRAMES_DIR = DATA_DIR / "init_frames"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = VIDEO_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading trained models...")
    # Load VQ-VAE model
    vq_vae = VQVAE(embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS).to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))

    # Load World Model based on argument
    world_model = None
    print(f"Loading GRU World Model from {WM_CHECKPOINT_FILENAME_GRU}")
    world_model = WorldModelGRU(
        latent_dim=VQVAE_EMBEDDING_DIM,
        codebook_size=VQVAE_NUM_EMBEDDINGS,
        action_dim=ACTION_DIM,
    ).to(DEVICE)

    # Compile the model to match the state_dict keys
    print("Compiling World Model...")
    world_model = torch.compile(world_model)

    world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))
    dream_function = dream_gru
    default_video_filename = "world_model_dream_gru.mp4"

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

    # --- setup for dreaming from a sequence of images ---
    image_files = sorted([os.path.join(INIT_FRAMES_DIR, f) for f in os.listdir(INIT_FRAMES_DIR)])
    priming_sequence = image_files[:10]

    # primed_h, initial_latent_tokens, initial_frame_tensor = get_starting_state_from_sequence(
    #     priming_sequence, world_model, vq_vae, DEVICE
    # )
    # For dreaming from a single image, use the first image in the sequence
    primed_h, initial_frame_tensor = get_starting_state_from_image(
        priming_sequence[0], world_model, vq_vae, DEVICE
    )

    initial_frame_np = initial_frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Generate the dream sequence
    dreamed_frames_list = dream_function(
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
        video_path = os.path.join(OUTPUT_DIR, args.output_filename if args.output_filename else default_video_filename)
        generate_video(dreamed_frames_list, video_path, fps=7)
