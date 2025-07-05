# play_game_sb3.py
import pathlib  # For creating paths
import time

import numpy as np
# Import from Stable Baselines3
from stable_baselines3 import PPO

# Import from local modules
from utils import (
    DEVICE, ENV_NAME, NUM_STACK,
    make_env_sb3  # Use the SB3 compatible environment creation function
)

# --- Configuration ---
NUM_EPISODES = 5  # How many episodes to play
PLAYBACK_SPEED_DELAY = 0  # Seconds to pause between steps
DETERMINISTIC_PLAY = True  # Use deterministic actions for playback

# --- Define Model Path ---
# SB3_MODEL_FILENAME = f"cnn_sb3_test_carracing-v3_final.zip"
SB3_MODEL_FILENAME = f"cnn_sb3_default_carracing-v3_final.zip"
# SB3_MODEL_FILENAME = f"cnn_sb3_default_carracing-v3_best/best_model.zip"  # best
# SB3_MODEL_FILENAME = f"cnn_sb3_default_carracing-v3/ppo_model_5000000_steps.zip" # specific step
SB3_MODEL_PATH = pathlib.Path("checkpoints") / SB3_MODEL_FILENAME


def play_sb3():
    print(f"Initializing environment: {ENV_NAME} with human rendering.")

    # --- Create Environment using make_env_sb3 ---
    # make_env_sb3 handles all necessary wrappers including PreprocessWrapper and ActionTransformWrapper
    # For playback, gamma for NormalizeReward wrapper doesn't strictly matter but use a sensible default.
    try:
        env = make_env_sb3(
            env_id=ENV_NAME,
            frame_stack_num=NUM_STACK,
            gamma=0.99,  # Standard gamma, used by NormalizeReward
            render_mode="human",
            max_episode_steps=1000,  # Typical for CarRacing
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

    # --- Play Episodes ---
    all_rewards = []
    for episode in range(NUM_EPISODES):
        print(f"\nStarting Episode {episode + 1}/{NUM_EPISODES}")

        # The observation from env.reset() will be the latent state z_stack_t
        obs_latent_state, info = env.reset()

        done = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not done and not truncated:
            # 1. Get Action from SB3 PPO Agent
            #    The observation (obs_latent_state) is already the processed latent state.
            #    The PPO agent expects this directly.
            action_from_agent, _states = ppo_agent.predict(obs_latent_state, deterministic=DETERMINISTIC_PLAY)

            # The 'action_from_agent' is in the space the agent was trained on,
            # which is Box([-1,1]^3) due to ActionTransformWrapper.
            # The ActionTransformWrapper inside 'env' will handle converting this
            # to CarRacing's native action space when env.step() is called.

            # 2. Step Environment
            obs_latent_state, reward, terminated, truncated, info = env.step(action_from_agent)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            if PLAYBACK_SPEED_DELAY > 0:
                time.sleep(PLAYBACK_SPEED_DELAY)

            # Rendering is handled by render_mode="human" in make_env_sb3

        print(f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward:.2f}")
        all_rewards.append(total_reward)

    # --- Cleanup ---
    if hasattr(env, 'close'):
        env.close()
    print("\nFinished playing.")
    if all_rewards:
        print(f"Average reward over {NUM_EPISODES} episodes: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":
    play_sb3()
