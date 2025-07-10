import gymnasium as gym
import torch
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from pathlib import Path
from PIL import Image
import numpy as np

from src.utils import preprocess_observation, DEVICE, VQ_VAE_CHECKPOINT_FILENAME
from src.vq_conv_vae import VQVAE

# --- Configuration ---
# You can change these parameters
MODEL_ID = "Pyro-X2/CarRacingSB3"
MODEL_FILENAME = "ppo-CarRacing-v3.zip"
NUM_IMAGES_TO_SAVE = 50  # Total number of sample images to generate
SAVE_EVERY_N_STEPS = 32  # Save a frame every N steps to get diverse images
OUTPUT_DIR = Path("../../assets/sample_images/")


def save_preprocessed_observation(preprocessed_obs, filename):
    """Saves a preprocessed (normalized, grayscale) observation as a PNG image."""
    # The observation is normalized in [0, 1], so scale it to [0, 255] for saving.
    img_array = (preprocessed_obs * 255).astype(np.uint8)
    # Squeeze the channel dimension (H, W, 1) -> (H, W) before creating the image.
    img = Image.fromarray(np.squeeze(img_array))
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
    step_count = 0
    reconstruction_data = []

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

                # save sample image
                image_filename = f"frame_{saved_count:04d}.png"
                image_path = OUTPUT_DIR / image_filename
                print(f"  - Saving preprocessed frame {saved_count + 1} to {image_path}")
                save_preprocessed_observation(preprocessed_obs, image_path)

                # save reconstructed image using VQ-VAE
                with torch.no_grad():
                    preprocessed_obs = torch.tensor(preprocessed_obs, dtype=torch.float32).permute(2, 0,
                                                                                                   1)
                    obs_tensor = preprocessed_obs.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to device
                    reconstructed_image, _, _, encoding_indices = vqvae_model(obs_tensor)
                    reconstructed_image = reconstructed_image.squeeze(0)
                    reconstructed_image = reconstructed_image.permute(1, 2, 0).cpu().numpy()
                    reconstructed_image = np.clip(reconstructed_image, 0, 1)  # Ensure values are in [0, 1]
                    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
                    reconstructed_image_filename = f"reconstructed_frame_{saved_count:04d}.png"
                    reconstructed_image_path = OUTPUT_DIR / reconstructed_image_filename
                    Image.fromarray(np.squeeze(reconstructed_image)).save(reconstructed_image_path)

                # Save the preprocessed observation for potential further analysis
                reconstruction_data.append({
                    "example_id": saved_count,
                    "original_image_path": "data/sample_images/" + image_filename,
                    "reconstructed_image_path": "data/sample_images/" + reconstructed_image_filename,
                    "token_grid": encoding_indices.cpu().numpy().tolist()
                })

                saved_count += 1

    # Save reconstruction data to a JSON file
    reconstruction_data_path = OUTPUT_DIR.parent / "reconstruction_data.json"
    print(f"Saving reconstruction data to {reconstruction_data_path}")
    with open(reconstruction_data_path, 'w') as f:
        import json
        json.dump(reconstruction_data, f, indent=4)

    env.close()


if __name__ == '__main__':
    main()
