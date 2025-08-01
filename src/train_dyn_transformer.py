import argparse
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Local imports
from src.impala_cnn import CustomCNN
from src.logger import ExperimentLogger
from src.train_ppo_sb3 import SB3_SAVE_DIR
from src.train_transformer_world_model import TransformerHistoryDataset, NUM_LOADER_WORKERS
from src.transformer_world_model import WorldModelTransformer, TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS, \
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT_RATE, TRANSFORMER_MAX_SEQ_LEN
from src.utils import (
    DEVICE, ENV_NAME, ACTION_DIM, NUM_STACK, VQ_VAE_CHECKPOINT_FILENAME,
    TRANSFORMER_WM_CHECKPOINTS_DIR, SB3_LOG_DIR, _create_dream_env, GRU_WM_CHECKPOINTS_DIR
)
from src.utils import _init_env_fn_sb3
from src.vq_conv_vae import VQVAE, GRID_SIZE, VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM
from src.world_model import WorldModelGRU, GRU_NUM_LAYERS, D_MODEL as GRU_D_MODEL


# --- Configuration ---
def get_combined_config(name="default"):
    """
    Combined configuration for Dyna-style training.
    Merges PPO and World Model configurations.
    """
    # --- World Model Config (Shared) ---
    wm_config = {
        "wm_epochs": 5,
        "wm_batch_size": 256,
        "wm_learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "grid_size": GRID_SIZE,
        "codebook_size": VQVAE_NUM_EMBEDDINGS,
        "vqvae_embed_dim": VQVAE_EMBEDDING_DIM,
        'num_loader_workers': NUM_LOADER_WORKERS,
    }

    # --- Transformer Specific Config ---
    transformer_config = {
        "history_length": 32,
        "embed_dim": TRANSFORMER_EMBED_DIM,
        "num_heads": TRANSFORMER_NUM_HEADS,
        "num_layers": TRANSFORMER_NUM_LAYERS,
        "ff_dim": TRANSFORMER_FF_DIM,
        "dropout_rate": TRANSFORMER_DROPOUT_RATE,
        "max_seq_len": TRANSFORMER_MAX_SEQ_LEN,
    }

    # --- GRU Specific Config ---
    gru_config = {
        "history_length": 1,  # GRU is recurrent, doesn't need long history dataset
        "gru_d_model": GRU_D_MODEL,
        "gru_num_layers": GRU_NUM_LAYERS,
    }

    # --- PPO Config ---
    ppo_config = {
        "policy": "CnnPolicy",
        "ppo_learning_rate": 45e-5,
        "n_steps": 1024,
        "ppo_batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "ppo_max_grad_norm": 0.5,
        "target_kl": 0.015,
        "policy_kwargs": dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=1024),
            net_arch=dict(pi=[256], vf=[256]),
            activation_fn=torch.nn.Tanh,
            log_std_init=-1.0,
            ortho_init=True,
        ),
    }

    # --- Dyna-style Training Loop Config ---
    dyna_config = {
        "total_real_steps": 1_000_000,
        "warmup_real_steps": 200_000,
        "wm_train_interval": 25_000,
        "dream_horizon": 15,
        "dream_steps_per_real_step": 1,
        "num_envs": 8,
        "num_dream_envs": 2,  # Number of parallel dream environments
        "max_episode_steps_collect": 1000,
        "seed": random.randint(0, 2 ** 31 - 1),
        "device": DEVICE,
        "action_dim": ACTION_DIM,
        "env_name": ENV_NAME,
    }

    # Combine configs
    combined_default = {**wm_config, **transformer_config, **gru_config, **ppo_config, **dyna_config}

    test_config = combined_default.copy()
    test_config.update({
        "total_real_steps": 2_000,
        "warmup_real_steps": 1000,
        "wm_train_interval": 500,
        "wm_epochs": 2,
        "wm_batch_size": 4,
        "num_loader_workers": 0,
        "history_length": 4,
        "dream_horizon": 5,
        "num_envs": 1,
        "n_steps": 128,
        "ppo_batch_size": 32,
        "policy_kwargs": dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[64], vf=[64]), activation_fn=torch.nn.Tanh,
        ),
    })

    configs = {"default": combined_default, "test": test_config}
    return configs[name]


@torch.no_grad()
def get_vq_indices(vq_vae_model, obs_batch, device) -> torch.Tensor:
    """
    Processes a batch of observations to get VQ-VAE token indices.

    This function handles both a single observation (H, W, C) and a
    batch of observations (B, H, W, C).

    Args:
        vq_vae_model: The pre-trained VQ-VAE model.
        obs_batch: A numpy array or tensor of observations.
                   Expected shape is (B, H, W, C) or (H, W, C).
        device: The torch device to use ('cuda' or 'cpu').

    Returns:
        A tensor of shape (B, N_tokens) containing the VQ-VAE indices.
    """
    # Convert numpy array to tensor and ensure it's on the correct device
    processed_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)

    # If a single observation (H, W, C) is passed, add a batch dimension
    if processed_tensor.ndim == 3:
        processed_tensor = processed_tensor.unsqueeze(0)  # Shape becomes (1, H, W, C)

    # Permute dimensions from (B, H, W, C) to (B, C, H, W) for Conv2D layers
    processed_tensor = processed_tensor.permute(0, 3, 1, 2)

    # Encode the batch to get the token indices from the VQ-VAE
    z_continuous = vq_vae_model.encoder(processed_tensor)
    z_continuous = vq_vae_model._pre_vq_conv(z_continuous)
    _, _, _, encoding_indices_out = vq_vae_model.vq_layer(z_continuous)

    # Get the batch size for reshaping the output
    batch_size = processed_tensor.size(0)

    # Flatten the grid of indices to a single vector per batch item -> (B, N_tokens)
    return encoding_indices_out.view(batch_size, -1)


class DynaCallback(BaseCallback):
    def __init__(self, trainer, verbose=0):
        super(DynaCallback, self).__init__(verbose)
        self.trainer = trainer

    def _on_step(self) -> bool:
        # This method is called after each step in the real environment
        # We use it to store transitions for the world model
        for i in range(self.model.n_envs):
            obs = self.locals['new_obs'][i]
            action = self.locals['actions'][i]
            reward = self.locals['rewards'][i]
            done = self.locals['dones'][i]
            info = self.locals['infos'][i]

            # Get tokens for prev and next states
            prev_tokens = get_vq_indices(self.trainer.vq_vae, self.model._last_obs[i][-1], self.trainer.device)
            next_tokens = get_vq_indices(self.trainer.vq_vae, obs[-1], self.trainer.device)

            is_first_step_val = bool(info.get('episode_start', done))
            is_done_val = bool(done or info.get("TimeLimit.truncated", False))

            self.trainer.replay_buffer.append({
                "prev_tokens": prev_tokens.squeeze(0).to(torch.int64).cpu(),
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([is_done_val], dtype=torch.float32),
                "next_tokens": next_tokens.squeeze(0).to(torch.int64).cpu(),
                "is_first_step": torch.tensor([is_first_step_val], dtype=torch.bool)
            })
        return True


class DynaTrainer:
    def __init__(self, config, config_name, world_model_type, wm_checkpoint_path=None, ppo_checkpoint=None,
                 run_name=None):
        self.config = config
        self.config_name = config_name
        self.world_model_type = world_model_type
        self.device = torch.device(config['device'])
        self.device_type = str(self.device).split(':')[0]
        seed = config.get('seed', None)
        print(f"Seed for this run: {seed}")
        set_random_seed(seed)

        print("Initializing models...")
        self.vq_vae = VQVAE().to(self.device)
        self.vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=self.device))
        self.vq_vae.eval()

        print(f"Initializing {self.world_model_type.upper()} World Model...")
        if self.world_model_type == 'transformer':
            self.world_model = WorldModelTransformer(
                embed_dim=config['embed_dim'], num_heads=config['num_heads'], num_layers=config['num_layers'],
                ff_dim=config['ff_dim'], grid_size=config['grid_size'], dropout_rate=config['dropout_rate'],
                max_seq_len=config['max_seq_len'], action_dim=config['action_dim'],
                codebook_size=config['codebook_size'], vqvae_embed_dim=config['vqvae_embed_dim']
            ).to(self.device)
            # print("Compiling Transformer World Model for performance...")
            # self.world_model = torch.compile(self.world_model)
        elif self.world_model_type == 'gru':
            self.world_model = WorldModelGRU(
                latent_dim=config['vqvae_embed_dim'],
                action_dim=config['action_dim'],
                d_model=config['gru_d_model'],
                gru_num_layers=config['gru_num_layers'],
                codebook_size=config['codebook_size'],
                grid_size=config['grid_size']
            ).to(self.device)
            # print("Compiling GRU World Model for performance...")
            # self.world_model = torch.compile(self.world_model)
        else:
            raise ValueError(f"Unknown world model type: {self.world_model_type}")

        if wm_checkpoint_path:
            print(f"Loading World Model from {wm_checkpoint_path}")
            self.world_model.load_state_dict(torch.load(wm_checkpoint_path, map_location=self.device))

        print("Initializing environments...")
        env_params = {
            "env_name_config": config["env_name"], "num_stack_config": NUM_STACK,
            "gamma_config": config["gamma"], "max_episode_steps_config": config["max_episode_steps_collect"],
        }
        if config["num_envs"] > 1:
            self.real_env = SubprocVecEnv([
                lambda i=i: _init_env_fn_sb3(rank=i, seed=config["seed"] + i, config_env_params=env_params)
                for i in range(config["num_envs"])
            ])
        else:
            self.real_env = DummyVecEnv(
                [lambda: _init_env_fn_sb3(rank=0, seed=config["seed"], config_env_params=env_params)])

        print("Initializing PPO agent...")
        self.ppo_agent = PPO(
            policy=config["policy"], env=self.real_env, learning_rate=config["ppo_learning_rate"],
            n_steps=config["n_steps"], batch_size=config["ppo_batch_size"], n_epochs=config["n_epochs"],
            gamma=config["gamma"], gae_lambda=config["gae_lambda"], clip_range=config["clip_range"],
            ent_coef=config["ent_coef"], vf_coef=config["vf_coef"], max_grad_norm=config["ppo_max_grad_norm"],
            target_kl=config["target_kl"], policy_kwargs=config["policy_kwargs"],
            tensorboard_log=str(SB3_LOG_DIR / f"dyn_{self.world_model_type}_{self.config_name}"),
            verbose=1, seed=config["seed"], device=self.device
        )

        if ppo_checkpoint:
            print(f"Loading PPO agent from {ppo_checkpoint}")
            self.ppo_agent.load(ppo_checkpoint, env=self.real_env)

        self.replay_buffer = deque(maxlen=config['total_real_steps'])
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=config['wm_learning_rate'])
        self.token_loss_fn = nn.CrossEntropyLoss()
        self.reward_loss_fn = nn.MSELoss()
        self.done_loss_fn = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler(enabled="cuda" in str(self.device))

        self.logger = ExperimentLogger(log_dir="logs", experiment_name=f"dyna_{self.world_model_type}_wm_training")
        if run_name is None:
            run_name = f"{self.config_name}_{int(time.time())}"
        self.logger.start_run(run_name=run_name, config=self.config)

    def train_wm(self, global_tqdm, global_step):
        global_tqdm.write(f"\n--- Training {self.world_model_type.upper()} World Model ---")
        history_len = self.config['history_length'] if self.world_model_type == 'transformer' else 1
        if len(self.replay_buffer) < history_len + 1:
            print("Not enough data for WM training. Skipping.")
            return

        self.world_model.train()
        buffer_list = list(self.replay_buffer)

        # For logging
        total_losses, token_losses, reward_losses, done_losses, grad_norms = [], [], [], [], []

        if self.world_model_type == 'transformer':
            data_dict = {
                'prev_tokens': torch.stack([s['prev_tokens'] for s in buffer_list]),
                'actions': torch.stack([s['action'] for s in buffer_list]),
                'rewards': torch.stack([s['reward'] for s in buffer_list]),
                'dones': torch.stack([s['done'] for s in buffer_list]),
                'next_tokens': torch.stack([s['next_tokens'] for s in buffer_list]),
                'is_first_steps': torch.stack([s['is_first_step'] for s in buffer_list]),
            }
            dataset = TransformerHistoryDataset(data_dict, self.config['history_length'])
            loader = DataLoader(dataset, batch_size=self.config['wm_batch_size'], shuffle=True,
                                num_workers=self.config['num_loader_workers'])

            for epoch in range(self.config['wm_epochs']):
                progress = tqdm(loader, desc=f"WM Epoch {epoch + 1}/{self.config['wm_epochs']}", position=0)
                for batch in progress:
                    for key in batch: batch[key] = batch[key].to(self.device)
                    with torch.amp.autocast(enabled="cuda" in str(self.device), device_type=self.device_type):
                        pred_logits, pred_reward, pred_done_logits, _ = self.world_model(
                            batch['action_history'], batch['latent_token_history']
                        )
                        b, h, w, c = pred_logits.shape
                        token_loss = self.token_loss_fn(pred_logits.view(b * h * w, c),
                                                        batch['target_next_tokens'].view(b * h * w))
                        reward_loss = self.reward_loss_fn(pred_reward, batch['target_reward'])
                        done_loss = self.done_loss_fn(pred_done_logits, batch['target_done'])
                        total_loss = token_loss + reward_loss + done_loss

                    self.wm_optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.wm_optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config['max_grad_norm'])
                    self.scaler.step(self.wm_optimizer)
                    self.scaler.update()
                    progress.set_postfix(loss=total_loss.item())

                    # Append metrics for logging
                    total_losses.append(total_loss.item())
                    token_losses.append(token_loss.item())
                    reward_losses.append(reward_loss.item())
                    done_losses.append(done_loss.item())
                    grad_norms.append(grad_norm.item())

        elif self.world_model_type == 'gru':
            # For GRU, we train on single steps (history is handled by hidden state)
            dataset = TensorDataset(
                torch.stack([s['prev_tokens'] for s in buffer_list]),
                torch.stack([s['action'] for s in buffer_list]),
                torch.stack([s['reward'] for s in buffer_list]),
                torch.stack([s['done'] for s in buffer_list]),
                torch.stack([s['next_tokens'] for s in buffer_list])
            )
            loader = DataLoader(dataset, batch_size=self.config['wm_batch_size'], shuffle=True,
                                num_workers=self.config['num_loader_workers'])

            for epoch in range(self.config['wm_epochs']):
                progress = tqdm(loader, desc=f"WM Epoch {epoch + 1}/{self.config['wm_epochs']}", position=0)
                for prev_tokens, actions, rewards, dones, next_tokens in progress:
                    prev_tokens, actions, rewards, dones, next_tokens = \
                        prev_tokens.to(self.device), actions.to(self.device), rewards.to(self.device), \
                            dones.to(self.device), next_tokens.to(self.device)

                    # Add sequence dimension (T=1)
                    prev_tokens = prev_tokens.unsqueeze(1)
                    actions = actions.unsqueeze(1)

                    # Get initial hidden state for the batch
                    initial_hidden_state = self.world_model.get_initial_hidden_state(
                        batch_size=prev_tokens.size(0), device=self.device
                    )

                    with torch.amp.autocast(enabled="cuda" in str(self.device), device_type=self.device_type):
                        pred_token_logits, pred_rewards, pred_dones, _, _ = self.world_model(
                            prev_tokens, actions, initial_hidden_state
                        )
                        # Reshape for loss calculation (remove sequence dimension)
                        pred_token_logits = pred_token_logits.squeeze(1)
                        pred_rewards = pred_rewards.squeeze(1)
                        pred_dones = pred_dones.squeeze(1)

                        token_loss = self.token_loss_fn(pred_token_logits.flatten(0, 1), next_tokens.flatten(0, 1))
                        reward_loss = self.reward_loss_fn(pred_rewards, rewards)
                        done_loss = self.done_loss_fn(pred_dones, dones)
                        total_loss = token_loss + reward_loss + done_loss

                    self.wm_optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.wm_optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config['max_grad_norm'])
                    self.scaler.step(self.wm_optimizer)
                    self.scaler.update()
                    progress.set_postfix(loss=total_loss.item())

                    # Append metrics for logging
                    total_losses.append(total_loss.item())
                    token_losses.append(token_loss.item())
                    reward_losses.append(reward_loss.item())
                    done_losses.append(done_loss.item())
                    grad_norms.append(grad_norm.item())

        if self.logger and total_losses:
            self.logger.log_metrics({
                'train/mean_total_loss': sum(total_losses) / len(total_losses),
                'train/mean_token_loss': sum(token_losses) / len(token_losses),
                'train/mean_reward_loss': sum(reward_losses) / len(reward_losses),
                'train/mean_done_loss': sum(done_losses) / len(done_losses),
                'train/mean_grad_norm': sum(grad_norms) / len(grad_norms),
            }, step=global_step)
        global_tqdm.write("--- World Model Training Finished ---")

    def train_agent_in_dream(self, global_tqdm):
        global_tqdm.write("\n--- Training Agent in Dream ---")
        if len(self.replay_buffer) < self.config.get('history_length', 1) + 1:
            global_tqdm.write("Not enough data to initialize dream. Skipping.")
            return

        wm_state_dict = self.world_model.state_dict()
        vq_vae_state_dict = self.vq_vae.state_dict()

        # Local variables for pickling
        config = self.config
        replay_buffer = self.replay_buffer
        world_model_type = self.world_model_type

        if self.config["num_dream_envs"] > 1:
            env_fns = [
                (lambda i=i: _create_dream_env(
                    config, wm_state_dict, vq_vae_state_dict, replay_buffer, config['seed'] + i,
                    world_model_type
                )) for i in range(self.config['num_dream_envs'])
            ]
            dream_env = SubprocVecEnv(env_fns, start_method='fork')
        else:
            dream_env = DummyVecEnv([lambda: _create_dream_env(
                self.config, wm_state_dict, vq_vae_state_dict, self.replay_buffer, self.config['seed'],
                self.world_model_type
            )])

        original_env = self.ppo_agent.get_env()
        self.ppo_agent.set_env(dream_env)
        dream_steps = self.config['wm_train_interval'] * self.config['dream_steps_per_real_step']
        self.ppo_agent.learn(total_timesteps=dream_steps, reset_num_timesteps=False, progress_bar=True)
        self.ppo_agent.set_env(original_env)
        self.ppo_agent.rollout_buffer.reset()
        dream_env.close()
        global_tqdm.write("--- Agent Dream Training Finished ---")

    def run(self):
        print("--- Starting Dyna-Style Training ---")
        callback = DynaCallback(trainer=self)
        total_real_steps = self.config['total_real_steps']
        wm_train_interval = self.config['wm_train_interval']
        wm_dir = TRANSFORMER_WM_CHECKPOINTS_DIR if self.world_model_type == 'transformer' else GRU_WM_CHECKPOINTS_DIR
        wm_dir.mkdir(exist_ok=True, parents=True)

        with tqdm(total=total_real_steps, desc="Total Real Steps", position=0) as pbar:
            while pbar.n < total_real_steps:
                last_steps = self.ppo_agent.num_timesteps
                steps_to_collect = min(wm_train_interval, total_real_steps - pbar.n)

                pbar.write(f"Collecting {steps_to_collect} real steps...")
                # Collect real data and train PPO on it
                self.ppo_agent.learn(
                    total_timesteps=steps_to_collect,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=True
                )
                # Update global progress bar
                steps_this_run = self.ppo_agent.num_timesteps - last_steps
                pbar.write(f"Collected {steps_this_run} steps.")
                pbar.update(steps_this_run)
                real_steps_collected = self.ppo_agent.num_timesteps

                # Train the World Model on the collected real data
                self.train_wm(tqdm, global_step=real_steps_collected)

                # Train the Agent in Dream (only after warmup)
                if real_steps_collected >= self.config['warmup_real_steps']:
                    self.train_agent_in_dream(tqdm)
                else:
                    pbar.write(
                        f"Warmup phase: Skipping dream training. ({real_steps_collected}/{self.config['warmup_real_steps']})")

                # Save models
                wm_path = wm_dir / f"dyn_{self.config_name}_wm_step_{real_steps_collected}.pth"
                ppo_path_dir = Path(
                    str(SB3_SAVE_DIR / f"dyn_{self.world_model_type}_{self.config_name}_{ENV_NAME.lower()}"))
                ppo_path_dir.mkdir(exist_ok=True, parents=True)
                ppo_path = ppo_path_dir / f"ppo_model_{real_steps_collected}_steps.zip"
                torch.save(self.world_model.state_dict(), wm_path)
                self.ppo_agent.save(ppo_path)
                pbar.write(f"Saved models at step {real_steps_collected}")

        self.real_env.close()
        if self.logger:
            self.logger.end_run()

        # Save models
        wm_path = wm_dir / f"dyn_{self.config_name}_{ENV_NAME.lower()}_wm_final.pth"
        ppo_path = str(SB3_SAVE_DIR / f"dyn_{self.world_model_type}_{self.config_name}_{ENV_NAME.lower()}_final.zip")
        torch.save(self.world_model.state_dict(), wm_path)
        print(f"World Model saved to {wm_path}")
        print(f"PPO Agent saved to {ppo_path}")
        self.ppo_agent.save(ppo_path)

        print("--- Dyna-Style Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a World Model with a PPO agent (Dyna-style).")
    parser.add_argument("--config", type=str, default="default", help="Configuration name ('default', 'test').")
    parser.add_argument("--world-model-type", type=str, default="transformer", choices=["transformer", "gru"],
                        help="Type of world model to use.")
    parser.add_argument("--wm-checkpoint", type=str, default=None,
                        help="Path to a pre-trained world model checkpoint.")
    parser.add_argument("--ppo-checkpoint", type=str, default=None, help="Path to a pre-trained PPO agent checkpoint.")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the logging run.")
    args = parser.parse_args()

    config = get_combined_config(args.config)
    trainer = DynaTrainer(config, config_name=args.config, world_model_type=args.world_model_type,
                          wm_checkpoint_path=args.wm_checkpoint,
                          ppo_checkpoint=args.ppo_checkpoint,
                          run_name=args.run_name)
    trainer.run()
