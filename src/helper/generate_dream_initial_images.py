import gymnasium as gym
import torch
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from pathlib import Path
from PIL import Image
import numpy as np

from src.utils import preprocess_observation, DEVICE, VQ_VAE_CHECKPOINT_FILENAME, DATA_DIR
from src.vq_conv_vae import VQVAE

# --- Configuration ---
# You can change these parameters
MODEL_ID = "Pyro-X2/CarRacingSB3"
MODEL_FILENAME = "ppo-CarRacing-v3.zip"
NUM_IMAGES_TO_SAVE = 10  # Total number of sample images to generate
OUTPUT_DIR = Path(DATA_DIR / "init_frames/")


def save_preprocessed_observation(preprocessed_obs, filename):
    """Saves a preprocessed observation as a PNG image."""
    # The observation is normalized in [0, 1], so scale it to [0, 255] for saving.
    img_array = (preprocessed_obs * 255).astype(np.uint8)

    # Adjust for grayscale or RGB images
    if len(img_array.shape) == 3 and img_array.shape[-1] == 1:  # Grayscale
        img_array = np.squeeze(img_array)  # Remove channel dimension
    img = Image.fromarray(img_array)
    img.save(filename)


def main():
    # --- Setup Directories ---
    print(f"Creating output directory at: {OUTPUT_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Pre-trained Model ---
    print(f"Loading model '{MODEL_ID}' from Hugging Face Hub...")
    try:
        # Load the model from Hugging Face Hub
        checkpoint = load_from_hub(MODEL_ID, MODEL_FILENAME)
        # Create the environment *without* human rendering to speed it up
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        # Load the model into the PPO class
        model = PPO.load(checkpoint, env=env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from Hugging Face Hub: {e}")
        return

    # --- load VQ-VAE model for reconstructing images ---
    vqvae_model = VQVAE().to(DEVICE)
    try:
        vqvae_model.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        print(f"VQ-VAE model loaded from '{VQ_VAE_CHECKPOINT_FILENAME}'.")
    except FileNotFoundError:
        print(f"VQ-VAE model file '{VQ_VAE_CHECKPOINT_FILENAME}' not found. Please train your model first.")
        return

    vqvae_model.eval()

    # --- Play and Collect Images ---
    saved_count = 0
    frame_count = 0  # Track the total number of frames in the episode

    # Run multiple episodes until we have enough images
    while saved_count < NUM_IMAGES_TO_SAVE:
        print(f"\nStarting new episode. Progress: {saved_count}/{NUM_IMAGES_TO_SAVE}")
        observation, info = env.reset()
        done = False
        frame_count = 0  # Reset frame count for the new episode

        while not done and saved_count < NUM_IMAGES_TO_SAVE:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frame_count += 1

            # Skip the first 50 frames
            if frame_count <= 50:
                continue

            # Skip every 4 frames
            if (frame_count - 50) % 4 != 0:
                continue

            # Apply the preprocessing function to the raw observation
            preprocessed_obs = preprocess_observation(observation)

            # Adjust for grayscale or RGB images
            if len(preprocessed_obs.shape) == 2:  # Grayscale
                preprocessed_obs = np.expand_dims(preprocessed_obs, axis=-1)  # Add channel dimension

            # Save sample image
            image_filename = f"frame_{saved_count:04d}.png"
            image_path = OUTPUT_DIR / image_filename
            print(f"  - Saving preprocessed frame {saved_count + 1} to {image_path}")
            save_preprocessed_observation(preprocessed_obs, image_path)

            saved_count += 1

    env.close()


if __name__ == '__main__':
    main()
