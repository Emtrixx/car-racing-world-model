# train_ppo_sb3.py
import time
import argparse
import os
import pathlib
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym  # Gymnasium is the new name for gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor  # For DummyVecEnv logging

# Import from local modules
from conv_vae import ConvVAE  # Assuming conv_vae.py is in the same directory or PYTHONPATH
from utils import (
    DEVICE, ENV_NAME, transform, VAE_CHECKPOINT_FILENAME,
    NUM_STACK, LATENT_DIM, ACTION_DIM,
    make_env_sb3, DEVICE_STR  # Use the new/modified make_env_sb3 from utils.py
)

print(f"Using device for main script: {DEVICE}")
SB3_SAVE_DIR = pathlib.Path("checkpoints")
SB3_LOG_DIR = pathlib.Path("logs")
SB3_SAVE_DIR.mkdir(parents=True, exist_ok=True)
SB3_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_config_sb3(name="default"):
    """
    Provides configurations for Stable Baselines3 PPO.
    Hyperparameters are adapted from the original PPO config.
    """
    configs = {
        "default": {
            # SB3 PPO Hyperparameters
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,  # Can be a schedule
            "n_steps": 2048,  # Corresponds to STEPS_PER_BATCH (per environment)
            "batch_size": 64,  # PPO's minibatch size
            "n_epochs": 10,  # Corresponds to EPOCHS_PER_UPDATE
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,  # Corresponds to INITIAL_ENTROPY_COEF (fixed for now)
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.015,  # For early stopping in PPO updates
            "sde_sample_freq": -1,  # Set to -1 to disable SDE for standard PPO

            # Policy keyword arguments for MlpPolicy
            "policy_kwargs": dict(
                # features_extractor_class=torch.nn.Identity, # Not needed if obs is already flat
                # features_extractor_kwargs=dict(features_dim=LATENT_DIM * NUM_STACK), # Not needed
                net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Matches Actor/Critic hidden layers
                activation_fn=torch.nn.Tanh,
                log_std_init=-1.0,  # Matches custom Actor's initial log_std bias
                ortho_init=True,  # SB3 default, can be False if issues arise
            ),

            # Training parameters
            "total_timesteps": 10_000_000,
            "num_envs": 4,  # Number of parallel environments
            "save_freq": 50_000,  # Timesteps, not updates. (orig: 50 updates * 2048 steps/update = 102400 steps)
            # Let's use a step-based frequency for SB3 CheckpointCallback
            "eval_freq": 20480,  # Timesteps per eval environment
            "n_eval_episodes": 5,
            "seed": 42,

            # Environment parameters (passed to make_env_sb3 via _init_env_fn_sb3)
            "env_name_config": ENV_NAME,
            "num_stack_config": NUM_STACK,
            "latent_dim_config": LATENT_DIM,
            "gamma_config": 0.99,  # For NormalizeReward wrapper
            "vae_checkpoint_path_config": VAE_CHECKPOINT_FILENAME,
            "max_episode_steps_config": 1000,  # Max steps per episode in CarRacing
        }
    }
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "total_timesteps": 6_000,
        "n_steps": 128,
        "num_envs": 2,
        "save_freq": 5_000,
        "eval_freq": 2048,
        "learning_rate": 3e-4,
    })
    return configs[name]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def _init_env_fn_sb3(rank: int, seed: int = 0, config_env_params: dict = None):
    """
    Creates an environment instance for SubprocVecEnv or DummyVecEnv.
    Each process/environment will call this function.
    """
    if config_env_params is None:
        config_env_params = {}

    set_random_seed(seed + rank)  # Ensure each environment has a different seed

    # TODO: check if main process is using the same device as subprocesses
    vae_device_for_subprocess = torch.device(DEVICE_STR)

    # print(f"Rank {rank}: Attempting to load VAE on device: {vae_device_for_subprocess}")

    # Ensure ConvVAE can be initialized with latent_dim, or adjust as per its definition
    vae_model = ConvVAE(latent_dim=config_env_params.get("latent_dim_config", LATENT_DIM)).to(vae_device_for_subprocess)
    vae_checkpoint_path = config_env_params.get("vae_checkpoint_path_config", VAE_CHECKPOINT_FILENAME)

    try:
        vae_model.load_state_dict(torch.load(vae_checkpoint_path, map_location=vae_device_for_subprocess))
        vae_model.eval()
        # print(f"Rank {rank}: Successfully loaded VAE from {vae_checkpoint_path} to {vae_device_for_subprocess}")
    except FileNotFoundError:
        print(f"Rank {rank}: ERROR: VAE checkpoint '{vae_checkpoint_path}' not found. Train VAE first.")
        raise
    except Exception as e:
        print(f"Rank {rank}: ERROR loading VAE: {e}")
        raise

    env = make_env_sb3(
        env_id=config_env_params.get("env_name_config", ENV_NAME),
        vae_model_instance=vae_model,
        transform_function=transform,  # Global transform from utils.py
        frame_stack_num=config_env_params.get("num_stack_config", NUM_STACK),
        single_latent_dim=config_env_params.get("latent_dim_config", LATENT_DIM),
        device_for_vae=vae_device_for_subprocess,
        gamma=config_env_params.get("gamma_config", 0.99),
        render_mode=config_env_params.get("render_mode", None),
        max_episode_steps=config_env_params.get("max_episode_steps_config", 1000),
        seed=seed + rank  # Pass seed to make_env_sb3 for its own seeding logic if any
    )
    # Monitor wrapper is important for SB3 to log episode rewards and lengths,
    # especially when using DummyVecEnv or if RecordEpisodeStatistics is not used inside make_env_sb3.
    env = Monitor(env)
    return env


def train_ppo_sb3(config_name: str):
    print(f"Starting Stable Baselines3 PPO training with config: {config_name}...")
    config = get_config_sb3(config_name)
    start_time = time.time()

    set_random_seed(config["seed"])

    # Prepare parameters for environment creation
    env_params_for_creation = {
        "env_name_config": config["env_name_config"],
        "num_stack_config": config["num_stack_config"],
        "latent_dim_config": config["latent_dim_config"],
        "gamma_config": config["gamma_config"],
        "vae_checkpoint_path_config": config["vae_checkpoint_path_config"],
        "max_episode_steps_config": config["max_episode_steps_config"],
        "num_envs_for_vae_device_check": config["num_envs"],  # Pass num_envs for VAE device heuristic
        # "render_mode": "human" if config["num_envs"] == 1 else None # Example for rendering
    }

    # Create vectorized environments
    if config["num_envs"] > 1:
        vec_env = SubprocVecEnv([
            lambda i=i: _init_env_fn_sb3(rank=i, seed=config["seed"], config_env_params=env_params_for_creation)
            for i in range(config["num_envs"])
        ])
    else:  # Use DummyVecEnv for single environment (easier debugging, but slower)
        vec_env = DummyVecEnv([
            lambda: _init_env_fn_sb3(rank=0, seed=config["seed"], config_env_params=env_params_for_creation)
        ])

    # Learning rate schedule
    lr_schedule = linear_schedule(config["learning_rate"])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config["save_freq"] // config["num_envs"], 1),  # Convert total steps to per-env steps
        save_path=str(SB3_SAVE_DIR / f"sb3_{config_name}_{ENV_NAME.lower()}"),
        name_prefix="ppo_model"
    )

    # Eval callback (optional, but good practice)
    # Create a separate evaluation environment (usually single, non-vectorized)
    eval_env_params = env_params_for_creation.copy()
    eval_env_params["num_envs_for_vae_device_check"] = 1  # Eval env is usually single
    eval_env = DummyVecEnv([lambda: _init_env_fn_sb3(rank=config["num_envs"], seed=config["seed"] + 1000,
                                                     config_env_params=eval_env_params)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SB3_SAVE_DIR / f"sb3_{config_name}_{ENV_NAME.lower()}_best"),
        log_path=str(SB3_LOG_DIR / f"sb3_{config_name}_{ENV_NAME.lower()}_eval"),
        eval_freq=max(config["eval_freq"] // config["num_envs"], 1),
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False
    )

    callbacks = [checkpoint_callback, eval_callback]

    # PPO should be run on CPU
    ppo_device = "cpu"

    # Create PPO model
    model = PPO(
        policy=config["policy"],
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        target_kl=config["target_kl"],
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=str(SB3_LOG_DIR / f"sb3_{config_name}_{ENV_NAME.lower()}"),
        verbose=1,
        seed=config["seed"],
        device=ppo_device  # SB3 will handle moving model to this device
    )

    print(f"PPO Model Device: {model.device}")
    print(f"Observation Space: {model.observation_space}")
    print(f"Action Space: {model.action_space}")

    # Train the agent
    print(f"Training for {config['total_timesteps']} total timesteps...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
    except Exception as e:
        print(f"Error during model.learn: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save the final model
        final_model_path = SB3_SAVE_DIR / f"sb3_{config_name}_{ENV_NAME.lower()}_final.zip"
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        vec_env.close()  # Important to close vectorized environments
        eval_env.close()

    total_time = time.time() - start_time
    print(f"Training finished. Total training time: {total_time:.2f} seconds")
    print(f"Models and logs saved in: {SB3_SAVE_DIR} and {SB3_LOG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent using Stable Baselines3.")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the configuration to use (e.g., 'default', 'test')."
    )
    args = parser.parse_args()


    train_ppo_sb3(args.config)

