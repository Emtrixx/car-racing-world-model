from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from play_game_sb3 import SB3_MODEL_PATH
from utils import (ENV_NAME, NUM_STACK, make_env_sb3,
                   VQ_VAE_CHECKPOINT_FILENAME, DEVICE)
from vq_conv_vae import VQVAE

# --- Video Configuration ---
NUM_EPISODES_TO_RECORD = 2  # How many episodes to record
VIDEO_FILENAME = f"videos/{ENV_NAME}_policy_visualization.mp4"
FPS = 7  # Frames per second for the output video
VIZ_WIDTH = 500  # Width of the policy visualization panel
UPSCALE_FACTOR = 1


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
    # --- Create Environment using make_env_sb3 ---
    # make_env_sb3 handles all necessary wrappers
    # For playback, gamma for NormalizeReward wrapper doesn't strictly matter but use a sensible default.
    print(f"Initializing environment: {ENV_NAME} for video recording.")
    try:
        env = make_env_sb3(
            env_id=ENV_NAME,
            frame_stack_num=NUM_STACK,
            gamma=0.99,  # Standard gamma, used by NormalizeReward
            render_mode="rgb_array",
            max_episode_steps=1000,
        )
        print("Environment created successfully with make_env_sb3.")
    except Exception as e:
        print(f"Error creating environment with make_env_sb3: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load Trained SB3 PPO Agent ---
    print(f"Loading trained SB3 PPO agent from: {SB3_MODEL_PATH}")
    if not SB3_MODEL_PATH.exists():
        print(f"ERROR: SB3 PPO Model not found at {SB3_MODEL_PATH}")
        if hasattr(env, 'close'): env.close()
        return
    try:
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=env)  # Provide env for action/obs space checks
        print(f"Successfully loaded SB3 PPO agent. Agent device: {ppo_agent.device}")
    except Exception as e:
        print(f"ERROR loading SB3 PPO agent: {e}")
        if hasattr(env, 'close'): env.close()
        import traceback
        traceback.print_exc()
        return

    # --- Video Writer Setup ---
    # Get sample frame for dimensions
    _obs_latent_state, _info = env.reset()  # obs_latent_state is used later by PPO agent

    sample_game_frame_rgb = env.render()  # Get a sample frame in RGB format

    # Get original dimensions
    original_game_height, original_game_width, _ = sample_game_frame_rgb.shape

    # Calculate new dimensions for the upscaled frame
    # Apply upscaling here
    game_width = original_game_width * UPSCALE_FACTOR
    game_height = original_game_height * UPSCALE_FACTOR

    # Viz height should match the *upscaled* game height
    viz_height = game_height

    combined_width = game_width + VIZ_WIDTH
    combined_height = game_height

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
        obs_latent_state, info = env.reset()  # obs is already a FrameStack
        done = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not done and not truncated:
            with torch.no_grad():
                obs_tensor = ppo_agent.policy.obs_to_tensor(obs_latent_state)[0]
                distribution = ppo_agent.policy.get_distribution(obs_tensor)

                # Get mean and stddev, and remove the batch dimension (which is 1)
                dist_mean = distribution.distribution.mean.squeeze(0)  # Shape will be (3,)
                dist_stddev = distribution.distribution.stddev.squeeze(0)  # Shape will be (3,)

                action_from_agent, _states = ppo_agent.predict(obs_latent_state, deterministic=True)

                obs_latent_state, reward, terminated, truncated, info = env.step(action_from_agent)
                done = terminated or truncated

            game_frame_rgb = env.render()  # Get the game frame in RGB format
            game_frame_bgr = cv2.cvtColor(game_frame_rgb, cv2.COLOR_RGB2BGR)
            game_frame_bgr = cv2.resize(game_frame_bgr, (game_width, game_height), interpolation=cv2.INTER_LINEAR)

            # Create Policy Visualization Frame
            viz_frame_bgr = create_policy_viz_frame(dist_mean, dist_stddev, VIZ_WIDTH, viz_height)

            # Ensure viz_frame has correct dimensions (it should if create_policy_viz_frame is correct)
            if viz_frame_bgr.shape[0] != game_height or viz_frame_bgr.shape[1] != VIZ_WIDTH:
                viz_frame_bgr = cv2.resize(viz_frame_bgr, (VIZ_WIDTH, game_height))

            # Combine Frames
            if game_frame_bgr.shape[0] == viz_frame_bgr.shape[0]:
                combined_frame_bgr = np.concatenate((game_frame_bgr, viz_frame_bgr), axis=1)
            else:  # Fallback if heights mismatch, though they shouldn't with current setup
                print(f"Warning: Frame height mismatch. Game: {game_frame_bgr.shape[0]}, Viz: {viz_frame_bgr.shape[0]}")
                # Create a placeholder viz of correct size to avoid crashing video writer
                placeholder_viz = np.zeros((game_height, VIZ_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder_viz, "Viz Error", (50, game_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                combined_frame_bgr = np.concatenate((game_frame_bgr, placeholder_viz), axis=1)

            # Write to Video
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
