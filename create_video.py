import gymnasium as gym
import torch
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

from utils import (DEVICE, ENV_NAME, transform,
                   VAE_CHECKPOINT_FILENAME, preprocess_and_encode_stack, FrameStackWrapper)
from utils_rl import PPO_ACTOR_SAVE_FILENAME  # Make sure this is correctly defined in utils_rl

from actor_critic import Actor  # Assuming Actor class is defined here or imported
from conv_vae import ConvVAE  # Assuming ConvVAE class is defined here or imported

# --- Video Configuration ---
NUM_EPISODES_TO_RECORD = 2  # How many episodes to record
VIDEO_FILENAME = f"videos/{ENV_NAME}_policy_visualization.mp4"
FPS = 30  # Frames per second for the output video
VIZ_WIDTH = 500  # Width of the policy visualization panel


# --- Matplotlib Visualization Function ---
def create_policy_viz_frame(dist_mean_tensor, dist_stddev_tensor, viz_width, viz_height):
    """
    Creates a visualization frame for the policy's action distributions.
    Args:
        dist_mean_tensor (torch.Tensor): Tensor of shape (3,) for mean of [steer, gas_raw, brake_raw]
        dist_stddev_tensor (torch.Tensor): Tensor of shape (3,) for stddev of [steer, gas_raw, brake_raw]
        viz_width (int): Width of the visualization image.
        viz_height (int): Height of the visualization image.
    Returns:
        np.array: BGR image of the visualization.
    """
    means = dist_mean_tensor.cpu().numpy()
    stds = dist_stddev_tensor.cpu().numpy()

    # Ensure stds are positive and non-zero for PDF calculation
    stds = np.maximum(stds, 1e-6)

    fig, axs = plt.subplots(3, 1, figsize=(viz_width / 100.0, viz_height / 100.0), dpi=100)
    action_names = ['Steering (raw)', 'Gas (raw)', 'Brake (raw)']
    # Raw action space for policy network outputs (typically around -1 to 1)
    # before specific environment scaling/transformations.
    raw_action_ranges = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)]  # Plotting range

    # Note on transformations for environment actions:
    # Steering: Typically tanh(raw_steer_mean) -> [-1, 1]
    # Gas/Brake: Typically (raw_gb_mean + 1.0) / 2.0 -> [0, 1]

    for i in range(3):
        mu, std = means[i], stds[i]
        x = np.linspace(raw_action_ranges[i][0], raw_action_ranges[i][1], 200)

        # PDF of a normal distribution
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)

        axs[i].plot(x, pdf, color='blue')
        axs[i].fill_between(x, pdf, alpha=0.3, color='blue')
        axs[i].set_title(f"{action_names[i]}\nMean: {mu:.2f}, Std: {std:.2f}")
        axs[i].set_xlim(raw_action_ranges[i])
        axs[i].set_ylim(bottom=0)  # Dynamic top based on PDF height
        if pdf.max() > 0:  # Avoid issues if pdf is all zeros (e.g. std is extremely small)
            axs[i].set_ylim(top=pdf.max() * 1.15)
        axs[i].grid(True, linestyle='--', alpha=0.6)

        # Indicate the actual mean value
        axs[i].axvline(mu, color='red', linestyle='--', label=f'Policy Mean: {mu:.2f}')

        # Add text about transformations
        transform_text = ""
        if i == 0:  # Steering
            transform_text = f"Env Action ≈ tanh({mu:.2f})"
        else:  # Gas/Brake
            transform_text = f"Env Action ≈ ({mu:.2f} + 1)/2"
        axs[i].text(0.02, 0.85, transform_text, transform=axs[i].transAxes, fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

    plt.tight_layout(pad=1.0)  # Adjust padding

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    viz_frame_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    plt.close(fig)  # Important to close the figure to free memory
    return viz_frame_bgr


def play_and_record():
    print(f"Initializing environment: {ENV_NAME} for video recording.")
    # Use render_mode="rgb_array" for video generation
    env = FrameStackWrapper(gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=1000))

    # --- Load Models ---
    print(f"Loading models to device: {DEVICE}")
    vae_model = ConvVAE().to(DEVICE)
    try:
        vae_model.load_state_dict(torch.load(VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vae_model.eval()
        print(f"Successfully loaded VAE: {VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VAE_CHECKPOINT_FILENAME}' not found.")
        env.close();
        return
    except Exception as e:
        print(f"ERROR loading VAE: {e}");
        env.close();
        return

    actor_model = Actor().to(DEVICE)
    try:
        actor_model.load_state_dict(torch.load(PPO_ACTOR_SAVE_FILENAME, map_location=DEVICE))
        actor_model.eval()
        print(f"Successfully loaded Actor: {PPO_ACTOR_SAVE_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: Actor checkpoint '{PPO_ACTOR_SAVE_FILENAME}' not found.")
        env.close();
        return
    except Exception as e:
        print(f"ERROR loading Actor: {e}");
        env.close();
        return

    # --- Video Writer Setup ---
    # Get sample frame for dimensions
    _ = env.reset()
    sample_game_frame_rgb = env.render()
    if sample_game_frame_rgb is None:
        print("Error: env.render() returned None. Cannot get frame dimensions.")
        env.close()
        return

    game_height, game_width, _ = sample_game_frame_rgb.shape
    viz_height = game_height  # Match game frame height for easy concatenation

    combined_width = game_width + VIZ_WIDTH
    combined_height = game_height

    # Define the codec and create VideoWriter object
    # Common codecs: 'mp4v' for .mp4, 'XVID' for .avi
    # 'X264' or 'H264' might offer better compression for MP4 if FFMPEG is properly installed.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FPS, (combined_width, combined_height))

    if not video_writer.isOpened():
        print(f"Error: VideoWriter failed to open for '{VIDEO_FILENAME}' with codec 'mp4v'. "
              "Ensure FFMPEG is installed or try a different codec (e.g., 'XVID' for an .avi file).")
        env.close()
        return
    print(f"Video recording setup complete. Output will be saved to: {VIDEO_FILENAME}")

    # --- Play Episodes and Record ---
    all_rewards = []
    for episode in range(NUM_EPISODES_TO_RECORD):
        print(f"\nStarting Episode {episode + 1}/{NUM_EPISODES_TO_RECORD}")
        obs, info = env.reset()  # obs is already a FrameStack
        done = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not done and not truncated:
            # 1. Preprocess and Encode Observation
            with torch.no_grad():
                z_stack_t = preprocess_and_encode_stack(obs, transform, vae_model, DEVICE)

                # 2. Get Action Distribution from Policy (Actor)
                dist = actor_model(z_stack_t.unsqueeze(0))  # Add batch dimension

                # These are the parameters we will visualize
                dist_mean = dist.mean.squeeze(0)  # Shape (3,) for [steer, gas_raw, brake_raw]
                dist_stddev = dist.stddev.squeeze(0)  # Shape (3,)

                # 3. Process Action for Environment Step (using mean for deterministic playback)
                action_raw_for_env = dist.mean  # Keep batch dim for processing

                action_steer = action_raw_for_env[:, :1]

                # Scale/shift gas/brake: raw [-1, 1] -> env [0, 1]
                action_processed_gb = (action_raw_for_env[:, 1:] + 1.0) / 2.0

                action_processed = torch.cat([action_steer, action_processed_gb], dim=1)

                # Clip to ensure bounds (important!)
                env_low = torch.tensor(env.action_space.low, device=DEVICE, dtype=torch.float32)
                env_high = torch.tensor(env.action_space.high, device=DEVICE, dtype=torch.float32)
                action_clipped = torch.clamp(action_processed, env_low, env_high)

                action_np = action_clipped.squeeze(0).cpu().numpy()

            # 4. Step Environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # 5. Render Game Frame
            game_frame_rgb = env.render()
            if game_frame_rgb is None:
                print("Warning: env.render() returned None during episode.")
                break
            game_frame_bgr = cv2.cvtColor(game_frame_rgb, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR

            # 6. Create Policy Visualization Frame
            viz_frame_bgr = create_policy_viz_frame(dist_mean, dist_stddev, VIZ_WIDTH, viz_height)

            # Ensure viz_frame has correct dimensions (it should if create_policy_viz_frame is correct)
            if viz_frame_bgr.shape[0] != game_height or viz_frame_bgr.shape[1] != VIZ_WIDTH:
                viz_frame_bgr = cv2.resize(viz_frame_bgr, (VIZ_WIDTH, game_height))

            # 7. Combine Frames
            if game_frame_bgr.shape[0] == viz_frame_bgr.shape[0]:
                combined_frame_bgr = np.concatenate((game_frame_bgr, viz_frame_bgr), axis=1)
            else:  # Fallback if heights mismatch, though they shouldn't with current setup
                print(f"Warning: Frame height mismatch. Game: {game_frame_bgr.shape[0]}, Viz: {viz_frame_bgr.shape[0]}")
                # Create a placeholder viz of correct size to avoid crashing video writer
                placeholder_viz = np.zeros((game_height, VIZ_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder_viz, "Viz Error", (50, game_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                combined_frame_bgr = np.concatenate((game_frame_bgr, placeholder_viz), axis=1)

            # 8. Write to Video
            video_writer.write(combined_frame_bgr)

            total_reward += reward
            step_count += 1

        print(f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward:.2f}")
        all_rewards.append(total_reward)

    # --- Cleanup ---
    env.close()
    video_writer.release()
    cv2.destroyAllWindows()  # If cv2.imshow was used
    print(f"\nFinished recording {NUM_EPISODES_TO_RECORD} episodes.")
    print(f"Video saved to: {VIDEO_FILENAME}")
    if all_rewards:
        print(f"Average reward over recorded episodes: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":

    play_and_record()