import argparse
import random
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.impala_cnn import CustomCNN
from src.transformer_world_model import (
    WorldModelTransformer,
    TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS, TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT_RATE, TRANSFORMER_MAX_SEQ_LEN
)
from src.utils import (
    DEVICE, ENV_NAME, ACTION_DIM, VQ_VAE_CHECKPOINT_FILENAME,
    _create_dream_env, SB3_CHECKPOINTS_DIR, SB3_LOG_DIR, _init_env_fn_sb3, NUM_STACK
)
from src.utils_wm import convert_dict_to_deque
from src.vq_conv_vae import VQVAE, GRID_SIZE, VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM
from src.world_model import WorldModelGRU, GRU_NUM_LAYERS, D_MODEL as GRU_D_MODEL
from src.train_transformer_world_model import HISTORY_LENGTH


def get_dream_config(name="default"):
    """
    Provides configurations for training PPO in the dream environment.
    Adapted from the Dyna-style training loop for consistency.
    """
    # --- World Model Config (Shared) ---
    wm_config = {
        "grid_size": GRID_SIZE,
        "codebook_size": VQVAE_NUM_EMBEDDINGS,
        "vqvae_embed_dim": VQVAE_EMBEDDING_DIM,
    }

    # --- Transformer Specific Config ---
    transformer_config = {
        "history_length": HISTORY_LENGTH,
        "embed_dim": TRANSFORMER_EMBED_DIM,
        "num_heads": TRANSFORMER_NUM_HEADS,
        "num_layers": TRANSFORMER_NUM_LAYERS,
        "ff_dim": TRANSFORMER_FF_DIM,
        "dropout_rate": TRANSFORMER_DROPOUT_RATE,
        "max_seq_len": TRANSFORMER_MAX_SEQ_LEN,
    }

    # --- GRU Specific Config ---
    gru_config = {
        "gru_d_model": GRU_D_MODEL,
        "gru_num_layers": GRU_NUM_LAYERS,
    }

    # --- PPO Config ---
    ppo_config = {
        "policy": "CnnPolicy",
        "ppo_learning_rate": 45e-5,
        "n_steps": 2048,  # Steps per batch (per environment)
        "ppo_batch_size": 1024,
        "n_epochs": 15,
        "gamma": 0.99,
        "gae_lambda": 0.9,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 1.0,
        "ppo_max_grad_norm": 0.8,
        "target_kl": 0.015,
        "eval_freq": 32_000,
        "n_eval_episodes": 5,
        "policy_kwargs": dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=1024),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            activation_fn=torch.nn.ReLU,
            log_std_init=-0.8,
            ortho_init=True,
        ),
    }

    # --- Dream Training Loop Config ---
    dream_config = {
        "total_timesteps": 2_000_000,
        "dream_horizon": 32,
        "num_envs": 8,
        "seed": random.randint(0, 2 ** 31 - 1),
        "device": DEVICE,
        "action_dim": ACTION_DIM,
        "env_name": ENV_NAME,
    }

    # Combine configs
    combined_default = {**wm_config, **transformer_config, **gru_config, **ppo_config, **dream_config}

    test_config = combined_default.copy()
    test_config.update({
        "total_timesteps": 20_000,
        "dream_horizon": 5,
        "num_envs": 2,
        "n_steps": 128,
        "ppo_batch_size": 256
    })

    configs = {"default": combined_default, "test": test_config}
    return configs[name]


def train_ppo_in_dream(config_name: str, model_type: str, wm_checkpoint_path: str, ppo_checkpoint_path: str,
                       buffer_path: str, run_name: str):
    """
    Train a PPO agent inside the world model's dream.
    """
    config = get_dream_config(config_name)
    seed = config.get('seed', None)
    print(f"Starting PPO training in dream with config: {config_name}, model: {model_type}, seed: {seed}...")
    set_random_seed(seed)
    start_time = time.time()

    # Load VQ-VAE
    print("Initializing VQ-VAE...")
    vq_vae = VQVAE().to(DEVICE)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
    vq_vae.eval()
    # vq_vae = torch.compile(vq_vae)

    # Load World Model
    print(f"Initializing {model_type.upper()} World Model...")
    if model_type == 'transformer':
        world_model = WorldModelTransformer(
            embed_dim=config['embed_dim'], num_heads=config['num_heads'], num_layers=config['num_layers'],
            ff_dim=config['ff_dim'], grid_size=config['grid_size'], dropout_rate=config['dropout_rate'],
            max_seq_len=config['max_seq_len'], action_dim=config['action_dim'],
            codebook_size=config['codebook_size'], vqvae_embed_dim=config['vqvae_embed_dim']
        ).to(DEVICE)
    elif model_type == 'gru':
        world_model = WorldModelGRU(
            latent_dim=config['vqvae_embed_dim'], action_dim=config['action_dim'],
            d_model=config['gru_d_model'], gru_num_layers=config['gru_num_layers'],
            codebook_size=config['codebook_size'], grid_size=config['grid_size']
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown world model type: {model_type}")

    world_model = torch.compile(world_model)
    print(f"Loading World Model from {wm_checkpoint_path}")
    world_model.load_state_dict(torch.load(wm_checkpoint_path, map_location=DEVICE))
    world_model.eval()

    # Load replay buffer
    print(f"Loading replay buffer from {buffer_path}")
    replay_buffer_data = torch.load(buffer_path, map_location='cpu')
    replay_buffer = convert_dict_to_deque(replay_buffer_data)
    print(f"Loaded {len(replay_buffer)} transitions from replay buffer.")

    # Create Dream Environment
    print("Creating dream environment...")
    wm_state_dict = {k: v.cpu() for k, v in world_model.state_dict().items()}
    vq_vae_state_dict = {k: v.cpu() for k, v in vq_vae.state_dict().items()}

    if config["num_envs"] > 1:
        env_fns = [
            (lambda i=i: _create_dream_env(
                config, wm_state_dict, vq_vae_state_dict, replay_buffer, config['seed'] + i,
                model_type
            )) for i in range(config['num_envs'])
        ]
        dream_env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        dream_env = DummyVecEnv([lambda: _create_dream_env(
            config, wm_state_dict, vq_vae_state_dict, replay_buffer, config['seed'],
            model_type
        )])

    # PPO Agent
    print("Initializing PPO agent...")
    if run_name is None:
        run_name = f"ppo_dream_{model_type}_{config_name}_{int(time.time())}"

    model = PPO(
        policy=config["policy"],
        env=dream_env,
        learning_rate=config["ppo_learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["ppo_batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["ppo_max_grad_norm"],
        target_kl=config.get("target_kl"),
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=str(SB3_LOG_DIR / run_name),
        verbose=1,
        seed=config["seed"],
        device=DEVICE,
    )

    if ppo_checkpoint_path:
        print(f"Loading PPO agent from {ppo_checkpoint_path}")
        model.load(ppo_checkpoint_path, env=dream_env)

    # Callbacks
    print("Setting up callbacks...")
    save_path_prefix = f"ppo_dream_{model_type}_{config_name}"
    eval_freq = max(config["eval_freq"] // config["num_envs"], 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=str(SB3_CHECKPOINTS_DIR / save_path_prefix),
        name_prefix="ppo_dream_model",
    )

    # Create real evaluation environment
    print("Creating real environment for evaluation...")
    eval_env_params = {
        "env_name_config": config["env_name"],
        "num_stack_config": NUM_STACK,
        "gamma_config": config["gamma"],
        "max_episode_steps_config": 1000,  # Standard eval length
    }
    eval_env = DummyVecEnv(
        [lambda: _init_env_fn_sb3(rank=config["num_envs"], seed=config["seed"] + 1000,
                                  config_env_params=eval_env_params)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SB3_CHECKPOINTS_DIR / f"{save_path_prefix}_best"),
        log_path=str(SB3_LOG_DIR / f"{save_path_prefix}_eval"),
        eval_freq=eval_freq,
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # Train
    print(f"Training for {config['total_timesteps']} total timesteps...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    finally:
        # Save the final model
        final_model_path = SB3_CHECKPOINTS_DIR / f"{save_path_prefix}_final.zip"
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        dream_env.close()
        eval_env.close()

    total_time = time.time() - start_time
    print(f"Training finished. Total training time: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent in a World Model's dream.")
    parser.add_argument("--config", type=str, default="default", help="Configuration name ('default', 'test').")
    parser.add_argument("--wm-type", type=str, required=True, choices=["transformer", "gru"],
                        help="Type of world model to use.")
    parser.add_argument("--wm-checkpoint", type=str, required=True,
                        help="Path to a pre-trained world model checkpoint.")
    parser.add_argument("--buffer", type=str, required=True, help="Path to a replay buffer file for initialization.")
    parser.add_argument("--ppo-checkpoint", type=str, default=None,
                        help="Path to a pre-trained PPO agent checkpoint to continue training.")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the logging run.")

    args = parser.parse_args()

    if "cuda" in str(DEVICE):
        print("Using float32 matmul high precision for CUDA training.")
        torch.set_float32_matmul_precision('high')

    train_ppo_in_dream(
        config_name=args.config,
        model_type=args.wm_type,
        wm_checkpoint_path=args.wm_checkpoint,
        ppo_checkpoint_path=args.ppo_checkpoint,
        buffer_path=args.buffer,
        run_name=args.run_name
    )
