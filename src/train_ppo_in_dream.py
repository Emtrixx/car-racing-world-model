import argparse
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.dream_env import GruDreamEnv
from src.dream_env_transformer import TransformerDreamEnv
from src.impala_cnn import CustomCNN
from src.transformer_world_model import WorldModelTransformer
from src.utils import (
    DEVICE,
    SB3_CHECKPOINTS_DIR,
    SB3_LOG_DIR,
    VQ_VAE_CHECKPOINT_BEST_FILENAME,
    WM_CHECKPOINT_FILENAME_GRU,
    DATA_DIR, FrameStackWrapper, WM_CHECKPOINT_FILENAME_TRANSFORMER,
)
from utils_vae import get_first_frame
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM
from src.world_model import WorldModelGRU
from src.utils import NUM_STACK, ACTION_DIM


def get_config_dream(name="default"):
    """
    Provides configurations for training PPO in the dream environment.
    Hyperparameters are adapted from the original PPO config.
    """
    configs = {
        "default": {
            # PPO Hyperparameters
            "policy": "CnnPolicy",
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            # Policy keyword arguments
            "policy_kwargs": dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=512),  # Test with a smaller feature dim
                net_arch=dict(pi=[64], vf=[64]),  # Test specific architecture
                activation_fn=torch.nn.Tanh,  # Inherited or explicitly set
                log_std_init=-1.0,  # Inherited or explicitly set
                ortho_init=True,  # Inherited or explicitly set
            ),
            # Training parameters
            "total_timesteps": 1_000_000,
        }
    }
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "total_timesteps": 6_000,  # Shorter for testing
        "n_steps": 512,  # Fewer steps for testing
    })
    return configs[name]


def train_ppo_in_dream(config_name: str, model_type: str, checkpoint_path: str = None):
    """
    Train a PPO agent inside the world model's dream.
    """
    print(f"Starting PPO training in dream with config: {config_name} and model: {model_type}...")
    start_time = time.time()

    # Load VQ-VAE
    vq_vae = VQVAE().to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_BEST_FILENAME, map_location=DEVICE))
    vq_vae.eval()

    # Load initial frame
    init_frames_dir = DATA_DIR / "init_frames"
    init_frame_files = list(init_frames_dir.glob("*.png"))
    if not init_frame_files:
        raise FileNotFoundError(f"No initial frames found in {init_frames_dir}")
    random_frame_file = random.choice(init_frame_files)
    initial_frame = get_first_frame(random_frame_file, vq_vae, device=DEVICE)

    # Create the Dream Environment based on the selected model type
    if model_type == "gru":
        world_model = WorldModelGRU(action_dim=ACTION_DIM, latent_dim=VQVAE_EMBEDDING_DIM).to(DEVICE)
        world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=DEVICE))
        world_model = torch.compile(world_model)
        env = GruDreamEnv(world_model, vq_vae, initial_frame)
    elif model_type == "transformer":
        world_model = WorldModelTransformer(action_dim=ACTION_DIM, vqvae_embed_dim=VQVAE_EMBEDDING_DIM).to(DEVICE)
        world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_TRANSFORMER, map_location=DEVICE))
        world_model = torch.compile(world_model)
        env = TransformerDreamEnv(world_model, vq_vae, initial_frame)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    world_model.eval()

    env = FrameStackWrapper(env, num_stack=NUM_STACK)  # Stack frames
    vec_env = DummyVecEnv([lambda: env])

    # PPO Configuration
    config = get_config_dream(config_name)

    # Callbacks
    save_path_prefix = f"ppo_dream_{model_type}_{config_name}"
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(SB3_CHECKPOINTS_DIR / save_path_prefix),
        name_prefix="ppo_dream_model",
    )
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=str(SB3_CHECKPOINTS_DIR / f"{save_path_prefix}_best"),
        log_path=str(SB3_LOG_DIR / f"{save_path_prefix}_eval"),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Create PPO model
    model = PPO(
        policy=config["policy"],
        env=vec_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=str(SB3_LOG_DIR / save_path_prefix),
        verbose=1,
        device=DEVICE,
    )

    if checkpoint_path:
        print(f"Loading pre-trained model from {checkpoint_path}...")
        model.load(checkpoint_path, env=vec_env)

    # Train the agent
    print(f"Training for {config['total_timesteps']} total timesteps...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save the final model
    final_model_path = SB3_CHECKPOINTS_DIR / f"{save_path_prefix}_final.zip"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    vec_env.close()

    total_time = time.time() - start_time
    print(f"Training finished. Total training time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent in a dream.")
    parser.add_argument(
        "--config", type=str, default="default", help="Configuration name."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gru",
        choices=["gru", "transformer"],
        help="Type of world model to use ('gru' or 'transformer')."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to a pre-trained model checkpoint."
    )
    args = parser.parse_args()

    train_ppo_in_dream(args.config, args.model_type, checkpoint_path=args.checkpoint)
