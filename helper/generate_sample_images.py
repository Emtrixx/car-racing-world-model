import gymnasium as gym
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from pathlib import Path
from PIL import Image
import numpy as np

from utils import preprocess_observation

# --- Configuration ---
# You can change these parameters
MODEL_ID = "Pyro-X2/CarRacingSB3"
MODEL_FILENAME = "ppo-CarRacing-v3.zip"
NUM_IMAGES_TO_SAVE = 50  # Total number of sample images to generate
SAVE_EVERY_N_STEPS = 32  # Save a frame every N steps to get diverse images
OUTPUT_DIR = Path("../assets/sample_images/")


def save_preprocessed_observation(preprocessed_obs, filename):
    """Saves a preprocessed (normalized, grayscale) observation as a PNG image."""
    # The observation is normalized in [0, 1], so scale it to [0, 255] for saving.
    img_array = (preprocessed_obs * 255).astype(np.uint8)
    # Squeeze the channel dimension (H, W, 1) -> (H, W) before creating the image.
    img = Image.fromarray(np.squeeze(img_array))
    img.save(filename)


def main():
    # --- 1. Setup Directories ---
    print(f"Creating output directory at: {OUTPUT_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Pre-trained Model ---
    print(f"Loading model '{MODEL_ID}' from Hugging Face Hub...")
    try:
        # Load the model from Hugging Face Hub
        checkpoint = load_from_hub(MODEL_ID, MODEL_FILENAME)
        # Create the environment *without* human rendering to speed it up
        env = gym.make("CarRacing-v3")
        # Load the model into the PPO class
        model = PPO.load(checkpoint, env=env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model from Hugging Face Hub: {e}")
        return

    # --- 3. Play and Collect Images ---
    saved_count = 0
    step_count = 0

    # Run multiple episodes until we have enough images
    while saved_count < NUM_IMAGES_TO_SAVE:
        print(f"\nStarting new episode. Progress: {saved_count}/{NUM_IMAGES_TO_SAVE}")
        observation, info = env.reset()
        done = False

        while not done and saved_count < NUM_IMAGES_TO_SAVE:
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            # Save frame at the specified interval
            if step_count % SAVE_EVERY_N_STEPS == 0:
                # Apply the preprocessing function to the raw observation
                preprocessed_obs = preprocess_observation(observation)

                image_path = OUTPUT_DIR / f"frame_{saved_count:04d}.png"
                print(f"  - Saving preprocessed frame {saved_count + 1} to {image_path}")
                save_preprocessed_observation(preprocessed_obs, image_path)
                saved_count += 1

    env.close()
    print(f"\nFinished. Saved {saved_count} sample images to '{OUTPUT_DIR.resolve()}'.")
    print("You can now use this folder as input for the `generate_assets.py` script.")


if __name__ == '__main__':
    main()
