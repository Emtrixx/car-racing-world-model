import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from tqdm import tqdm
import base64
import io
import requests
from PIL import Image
from pydantic import ValidationError

from src.play_game_sb3 import SB3_MODEL_PATH
from src.train_dyna_loop import get_vq_indices
from src.utils import \
    DEVICE, \
    VQ_VAE_CHECKPOINT_FILENAME, \
    ENV_NAME, \
    WM_CHECKPOINT_FILENAME_GRU, \
    make_env_sb3, \
    NUM_STACK, \
    ACTION_DIM, DATA_DIR
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM
from src.world_model import WorldModelGRU, D_MODEL, GRU_NUM_LAYERS

# --- Configuration ---
OLLAMA_API_URL = "http://192.168.2.39:11435/api/generate"
MODEL_NAME = "gemma3:12b"
ONE_SHOT_IMAGE_PATH = DATA_DIR / "one_shot_example.png"

# Your well-crafted prompt
PROMPT_TEXT = """
Your Role: You are an expert F1 racing analyst and data labeler. Your task is to analyze a single top-down image from a racing simulator and output a precise, quantitative state description in a JSON format.

Input: A single image frame showing a red race car on a track.

Output Instructions: Analyze the provided image and return only a single JSON object with the exact structure and fields defined below. Do not include any explanatory text, markdown formatting, or anything other than the JSON object itself.

JSON Schema and Field Definitions
Output Structure:

{
  "car_state": {
    "position_normalized": 0.0,
    "angle_relative_to_track_deg": 0.0,
    "wheels_off_track": 0
  },
  "track_state": {
    "curvature_current": 0.0,
    "curvature_upcoming": 0.0,
    "distance_to_next_turn_m": 0.0
  }
}
Field Definitions:

car_state:

position_normalized (Number): The car's lateral position, where 0.0 is the track's centerline. -1.0 is the center of the car on the track's left edge, and 1.0 is the right edge. Crucially, if the car is off-track, this value should extrapolate beyond this range. For example, a car halfway into the grass on the right would be 1.5. The "track" includes the red-and-white kerbs.

angle_relative_to_track_deg (Number): The car's angle in degrees relative to the track's centerline. 0.0 is perfectly aligned. A positive value means the car is pointing to the right of the track's direction (e.g., 15.0). A negative value means it is pointing to the left (e.g., -10.0).

wheels_off_track (Integer): The number of wheels (0, 1, 2, 3, or 4) that are on the green grass area.

track_state:

curvature_current (Number): The curvature of the track segment the car is currently on, from -1.0 (sharpest left turn) to 1.0 (sharpest right turn). A straight section is 0.0.

curvature_upcoming (Number): The curvature of the next track segment visible ahead of the car. Use the same -1.0 to 1.0 scale. If the track ahead is straight, this should be 0.0.

distance_to_next_turn_m (Number): Your best estimate of the distance in meters to the beginning of the next turn or the end of the current one. Assume the car is about 4 meters long for scale.

Example (One-Shot Learning)
Input Image:
[This is where the first image you send will be placed]

Required Output:

{
  "car_state": {
    "position_normalized": -0.7,
    "angle_relative_to_track_deg": -8.0,
    "wheels_off_track": 0
  },
  "track_state": {
    "curvature_current": -0.7,
    "curvature_upcoming": 0.0,
    "distance_to_next_turn_m": 10.0
  }
}
Your Task
Now, analyze the following new image and provide the corresponding JSON output.
"""

from pydantic import BaseModel, Field


class CarState(BaseModel):
    """
    Defines the state of the car at a specific frame.
    """
    position_normalized: float = Field(
        ...,
        description=(
            "The car's lateral position, where 0.0 is the track's centerline. "
            "-1.0 is the left edge, 1.0 is the right edge. Can extrapolate beyond this range."
        )
    )
    angle_relative_to_track_deg: float = Field(
        ...,
        description="The car's angle in degrees relative to the track's centerline. Positive is right, negative is left."
    )
    wheels_off_track: int = Field(
        ...,
        ge=0,
        le=4,
        description="The number of wheels (0, 1, 2, 3, or 4) that are on the grass."
    )


class TrackState(BaseModel):
    """
    Defines the state of the track relevant to the car.
    """
    curvature_current: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Curvature of the current track segment from -1.0 (sharpest left) to 1.0 (sharpest right)."
    )
    curvature_upcoming: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Curvature of the next visible track segment using the same -1.0 to 1.0 scale."
    )
    distance_to_next_turn_m: float = Field(
        ...,
        description="Estimated distance in meters to the beginning of the next turn."
    )


class F1FrameData(BaseModel):
    """
    The root JSON object for a single frame analysis of an F1 car.
    """
    car_state: CarState
    track_state: TrackState


def image_to_base64(image_array: np.ndarray) -> str:
    """Converts a NumPy array image to a Base64 encoded string."""
    # Assumes the input array is in a format that PIL can understand (e.g., uint8, RGB)
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_label(image_array: np.ndarray, one_shot_image_base64: str) -> dict | None:
    """
    Sends an image and a text prompt to a local Ollama VLM,
    gets a JSON response, validates it, and returns it as a dictionary.
    """
    print("Encoding image...")
    base64_image = image_to_base64(image_array)

    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_TEXT,
        "images": [one_shot_image_base64, image_to_label_base64],
        "format": "json",  # Crucial for forcing JSON output
        "stream": False  # Get the full response at once
    }

    try:
        print("Sending request to Ollama...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_content = response.json().get('response')
        if not response_content:
            print("Error: Received an empty response from the model.")
            return None

        # The model's response is a string, which we need to parse into JSON
        model_output = json.loads(response_content)

        print("Validating JSON response against Pydantic schema...")
        validated_data = F1FrameData.model_validate(model_output)

        return validated_data.model_dump()

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from model response.")
        print(f"Raw response: {response.json().get('response')}")
        return None
    except ValidationError as e:
        print("Error: Model output failed Pydantic validation.")
        print(e)
        return None


def analyze_latent_space(
        ppo_path: str,
        vq_vae_path: str,
        wm_path: str,
        output_dir: str,
        num_rollouts: int,
        steps_per_rollout: int
):
    """
    Generates rollouts, extracts latent states from a GRU World Model,
    and labels frames using a multi-modal model.
    """
    print(f"Using device: {DEVICE}")

    # --- Load and encode the one-shot example image ---
    print(f"Loading one-shot example image from '{ONE_SHOT_IMAGE_PATH}'...")
    try:
        one_shot_image = cv2.imread(str(ONE_SHOT_IMAGE_PATH))
        if one_shot_image is None:
            raise FileNotFoundError(f"Could not load image from {ONE_SHOT_IMAGE_PATH}")
        one_shot_image_rgb = cv2.cvtColor(one_shot_image, cv2.COLOR_BGR2RGB)
        one_shot_base64 = image_to_base64(one_shot_image_rgb)
        print("One-shot image loaded and encoded successfully.")
    except (FileNotFoundError, cv2.error) as e:
        print(f"Fatal Error: Could not process the one-shot example image. {e}")
        return

    # --- Setup Directories ---
    base_path = Path(output_dir)
    rollouts_path = base_path / "rollouts"
    rollouts_path.mkdir(parents=True, exist_ok=True)
    metadata_path = base_path / "metadata.jsonl"

    # --- Load Models ---
    print("Loading models (VQ-VAE, PPO Agent, and GRU World Model)...")
    env = make_env_sb3(env_id=ENV_NAME, frame_stack_num=NUM_STACK)

    vq_vae = VQVAE().to(DEVICE)
    vq_vae.load_state_dict(torch.load(vq_vae_path, map_location=DEVICE))
    vq_vae.eval()

    ppo_agent = PPO.load(ppo_path, device=DEVICE, env=env)

    world_model = WorldModelGRU(
        latent_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        gru_num_layers=GRU_NUM_LAYERS
    ).to(DEVICE)
    world_model = torch.compile(world_model)
    world_model.load_state_dict(torch.load(wm_path, map_location=DEVICE))
    world_model.eval()
    print("All models loaded successfully.")

    # --- Main Loop ---
    with open(metadata_path, 'w') as metadata_file:
        for rollout_id in range(num_rollouts):
            print(f"--- Starting Rollout {rollout_id + 1}/{num_rollouts} ---")
            rollout_frames_path = rollouts_path / f"rollout_{rollout_id:03d}"
            rollout_frames_path.mkdir(exist_ok=True)

            obs, _ = env.reset()
            hidden_state = world_model.get_initial_hidden_state(1, DEVICE)

            # Use the last frame of the stack for VQ-VAE and labeling
            latest_frame = obs[-1]

            # Initial action (dummy)
            action = np.zeros(env.action_space.shape)

            for step in tqdm(range(steps_per_rollout), desc="Generating steps"):
                # --- 1. Get Latent State from World Model ---
                with torch.no_grad():
                    # Get VQ-VAE tokens for the current observation
                    obs_tokens = get_vq_indices(vq_vae, latest_frame, DEVICE).unsqueeze(0)

                    # Prepare action tensor
                    action_tensor = torch.from_numpy(action).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

                    # Get the next hidden state from the GRU
                    _, _, _, hidden_state, _ = world_model(obs_tokens, action_tensor, hidden_state)

                # The latent state is the final hidden state from the GRU
                latent_state_np = hidden_state.squeeze().cpu().numpy()

                # --- 2. Save Frame and Get Label ---
                frame_path = rollout_frames_path / f"frame_{step:04d}.png"

                frame_rgb = (latest_frame * 255).astype(np.uint8)
                # Get label from the multi-modal model
                label = get_label(frame_rgb, one_shot_base64)

                print(f"Label for step {step}: {label}")

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Convert to RGB for saving and labeling
                cv2.imwrite(str(frame_path), frame_bgr)

                # --- 3. Step Environment ---
                action, _ = ppo_agent.predict(obs, deterministic=False)
                next_obs, reward, done, truncated, info = env.step(action)

                # --- 4. Store Metadata ---
                metadata = {
                    "rollout_id": rollout_id,
                    "step": step,
                    "frame_path": str(frame_path.relative_to(base_path)),
                    "action": action.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "latent_state": latent_state_np.tolist(),
                    "label": label
                }
                metadata_file.write(json.dumps(metadata) + '\n')

                obs = next_obs
                latest_frame = obs[-1]

                if done or truncated:
                    print(f"Rollout ended early at step {step}.")
                    break

    env.close()
    print(f"\nAnalysis complete. Data saved to '{base_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the latent space of a GRU World Model by generating and labeling rollouts."
    )
    parser.add_argument(
        "--output-dir", type=str, default=DATA_DIR / "gru_latent_analysis",
        help="Directory to save the output data."
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=10,
        help="Number of rollouts to generate."
    )
    parser.add_argument(
        "--steps-per-rollout", type=int, default=250,
        help="Maximum number of steps per rollout."
    )
    parser.add_argument(
        "--ppo-path", type=str,
        default=SB3_MODEL_PATH,
        help="Path to the PPO agent checkpoint."
    )
    parser.add_argument(
        "--vq-vae-path", type=str, default=VQ_VAE_CHECKPOINT_FILENAME,
        help="Path to the VQ-VAE checkpoint."
    )
    parser.add_argument(
        "--wm-path", type=str, default=WM_CHECKPOINT_FILENAME_GRU,
        help="Path to the GRU World Model checkpoint."
    )
    args = parser.parse_args()

    analyze_latent_space(
        ppo_path=args.ppo_path,
        vq_vae_path=args.vq_vae_path,
        wm_path=args.wm_path,
        output_dir=args.output_dir,
        num_rollouts=args.num_rollouts,
        steps_per_rollout=args.steps_per_rollout
    )
