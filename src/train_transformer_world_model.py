import argparse
import multiprocessing as mp
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from src.logger import ExperimentLogger
from src.play_game_sb3 import SB3_MODEL_PATH
from src.transformer_world_model import WorldModelTransformer, TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS, \
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT_RATE
from src.utils import (
    ENV_NAME,
    ACTION_DIM,
    DEVICE, VQ_VAE_CHECKPOINT_FILENAME,
    make_env_sb3, NUM_STACK, TRANSFORMER_WM_CHECKPOINTS_DIR
)
from src.vq_conv_vae import GRID_SIZE
from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM, VQVAE

# --- Configuration ---
# Training Hyperparameters
NUM_STEPS = 1_000_000
WM_EPOCHS = 10
WM_BATCH_SIZE = 32
WM_LEARNING_RATE = 1e-4
HISTORY_LENGTH = 16  # Renamed from SEQUENCE_LENGTH
MAX_GRAD_NORM = 1.0

# Parallelism Configuration
NUM_COLLECTION_WORKERS = 4
NUM_LOADER_WORKERS = 4

# Environment settings for data collection
MAX_EPISODE_STEPS_COLLECT = 1000


def get_config(name="default"):
    """Gets the configuration dictionary for training."""
    configs = {
        "default": {
            "env_name": ENV_NAME,
            'action_dim': ACTION_DIM,
            'num_steps': NUM_STEPS,
            'epochs': WM_EPOCHS,
            'learning_rate': WM_LEARNING_RATE,
            'batch_size': WM_BATCH_SIZE,
            'history_length': HISTORY_LENGTH,
            'max_grad_norm': MAX_GRAD_NORM,
            'max_episode_steps_collect': MAX_EPISODE_STEPS_COLLECT,
            'device': DEVICE,
            'num_collection_workers': NUM_COLLECTION_WORKERS,
            'num_loader_workers': NUM_LOADER_WORKERS,
            "validation_split": 0.1,
            "random_seed": random.randint(0, 2 ** 31 - 1),
            "val_freq": 200,
            "embed_dim": TRANSFORMER_EMBED_DIM,
            "num_heads": TRANSFORMER_NUM_HEADS,
            "num_layers": TRANSFORMER_NUM_LAYERS,
            "ff_dim": TRANSFORMER_FF_DIM,
            "grid_size": GRID_SIZE,
            "dropout_rate": TRANSFORMER_DROPOUT_RATE,
            # It's the length of the history multiplied by (tokens per state + 1 for action).
            "max_seq_len": 1024  # for positional encoding
        }
    }
    # Test configuration for quick runs
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "num_steps": 500,
        "epochs": 3,
        "batch_size": 4,
        "history_length": 4,  # Shorter history for testing
        "num_collection_workers": 2,
        "num_loader_workers": 2,
        "max_episode_steps_collect": 100,
        "dropout_rate": 0.1,
    })
    return configs[name]


class WorldModelDataCollector:
    """Collects training data for the World Model by running a pretrained policy."""

    def __init__(self, env, ppo_agent, vq_vae_model, device):
        self.env = env
        self.ppo_agent = ppo_agent
        self.vq_vae_model = vq_vae_model
        self.device = device
        self.replay_buffer = deque(maxlen=2_000_000)  # Increased buffer size

    def get_vq_indices(self, obs_raw_numpy: np.ndarray) -> torch.Tensor:
        """
        Helper function to preprocess a raw observation and get VQ-VAE token indices.

        Args:
            obs_raw_numpy (np.ndarray): A single raw frame from the environment.

        Returns:
            torch.Tensor: A tensor of token indices with shape [1, 16].
        """
        # Preprocess the raw observation
        # processed_frame = preprocess_observation(obs_raw_numpy)
        # env now returns preprocessed frames directly
        processed_frame = obs_raw_numpy

        # Convert to tensor, add batch dim, and send to device
        processed_tensor = torch.tensor(processed_frame, dtype=torch.float32, device=self.device)
        processed_tensor = processed_tensor.permute(2, 0, 1)  # to CHW
        processed_tensor = processed_tensor.unsqueeze(0)  # to BCHW

        # Encode to get the token indices from the VQ-VAE
        with torch.no_grad():
            z_continuous = self.vq_vae_model.encoder(processed_tensor)
            z_continuous = self.vq_vae_model._pre_vq_conv(z_continuous)
            _loss, _quantized_out, _perplexity, encoding_indices_out = self.vq_vae_model.vq_layer(z_continuous)

        return encoding_indices_out.view(1, -1)  # Flatten to [1, 16]

    def collect_steps(self, num_steps: int):
        """Runs the PPO agent in the environment for a given number of steps."""
        print(f"Collecting {num_steps} steps of experience...")
        obs, _ = self.env.reset()

        # We need the initial state's tokens to start the buffer correctly.
        initial_tokens = self.get_vq_indices(obs[-1])

        for step in tqdm(range(num_steps), desc="Collecting Steps"):
            action, _ = self.ppo_agent.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, info = self.env.step(action)
            next_state_tokens = self.get_vq_indices(next_obs[-1])

            self.replay_buffer.append({
                "prev_tokens": initial_tokens.squeeze(0).to(torch.int64),
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done or truncated], dtype=torch.float32),
                "next_tokens": next_state_tokens.squeeze(0).to(torch.int64)
            })

            initial_tokens = next_state_tokens
            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()
                initial_tokens = self.get_vq_indices(obs[-1])


def collect_sequences_worker(worker_id, num_steps_to_collect, env_name, device_str, max_episode_steps):
    """Worker function for parallel data collection."""
    try:
        print(f"[Worker {worker_id}] Starting...")
        env = make_env_sb3(env_id=env_name, frame_stack_num=NUM_STACK, max_episode_steps=max_episode_steps)
        ppo_agent = PPO.load(SB3_MODEL_PATH, device=device_str, env=env)
        vq_vae = VQVAE().to(device_str)
        vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=device_str))
        vq_vae.eval()

        collector = WorldModelDataCollector(env, ppo_agent, vq_vae, device_str)
        collector.collect_steps(num_steps=num_steps_to_collect)
        env.close()

        print(f"[Worker {worker_id}] Finished. Collected {len(collector.replay_buffer)} transitions.")

        # Package data for returning
        data_to_save = {
            'prev_tokens': torch.stack([s['prev_tokens'] for s in collector.replay_buffer]),
            'actions': torch.stack([s['action'] for s in collector.replay_buffer]),
            'rewards': torch.stack([s['reward'] for s in collector.replay_buffer]),
            'dones': torch.stack([s['done'] for s in collector.replay_buffer]),
            'next_tokens': torch.stack([s['next_tokens'] for s in collector.replay_buffer]),
        }

        temp_dir = Path("./tmp_worker_data")
        temp_dir.mkdir(exist_ok=True)
        filepath = temp_dir / f"temp_data_{worker_id}_{int(time.time() * 1000)}.pt"
        torch.save(data_to_save, filepath)
        return str(filepath)

    except Exception as e:
        print(f"[Worker {worker_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_data_parallel(num_steps, device, num_workers, env_name, max_episode_steps):
    """Manages parallel data collection and aggregates results."""
    print(f"Starting parallel collection with {num_workers} workers for {num_steps} total steps...")
    if num_workers <= 0:
        num_workers = 1

    steps_per_worker = np.full(num_workers, num_steps // num_workers)
    steps_per_worker[:num_steps % num_workers] += 1

    worker_args = [(i, steps_per_worker[i], env_name, device, max_episode_steps) for i in range(num_workers)]

    with mp.Pool(processes=num_workers) as pool:
        filepaths = pool.starmap(collect_sequences_worker, worker_args)

    all_data = []
    for path in filepaths:
        if path and Path(path).exists():
            try:
                all_data.append(torch.load(path))
                os.remove(path)
            except Exception as e:
                print(f"Error loading or removing temp file {path}: {e}")

    if not all_data:
        return None

    # Concatenate data from all workers
    final_buffer = {key: torch.cat([d[key] for d in all_data], dim=0) for key in all_data[0]}
    print(f"Finished collecting. Total transitions: {len(final_buffer['actions'])}.")
    return final_buffer


class TransformerHistoryDataset(Dataset):
    """Dataset to provide history of states and actions for the world model."""

    def __init__(self, data_dict, history_length):
        self.data = data_dict
        # History_length is the number of (state, action) pairs
        self.history_length = history_length
        # We need history_length transitions to form one sequence, plus one more for the target
        self.num_sequences = len(data_dict['actions']) - history_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # History ends at idx + history_length. Target is the sample at this index.
        end_idx = idx + self.history_length

        # History of actions and previous states
        action_history = self.data['actions'][idx:end_idx]
        token_history = self.data['prev_tokens'][idx:end_idx]

        # The target is the outcome of the last action in the history
        target_next_tokens = self.data['next_tokens'][end_idx - 1]
        target_reward = self.data['rewards'][end_idx - 1]
        target_done = self.data['dones'][end_idx - 1]

        return {
            'action_history': action_history,
            'latent_token_history': token_history,
            'target_next_tokens': target_next_tokens,
            'target_reward': target_reward,
            'target_done': target_done,
        }


class WorldModelTransformerTrainer:
    """Trainer for the history-aware Transformer World Model."""

    def __init__(self, world_model, vq_vae_model, config, train_dataloader, val_dataloader=None, logger=None):
        self.world_model = world_model
        self.vq_vae_model = vq_vae_model
        self.config = config
        self.device = config['device']
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger

        self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=config['learning_rate'])
        self.token_loss_fn = nn.CrossEntropyLoss()
        self.reward_loss_fn = nn.MSELoss()
        self.done_loss_fn = nn.BCEWithLogitsLoss()

    def _run_batch(self, batch, is_train=True):
        """Runs a single batch through the model and computes loss."""
        for key in batch:
            batch[key] = batch[key].to(self.device)

        action_hist = batch['action_history']
        token_hist = batch['latent_token_history']
        target_tokens = batch['target_next_tokens']
        target_reward = batch['target_reward']
        target_done = batch['target_done']

        pred_logits, pred_reward, pred_done_logits, _ = self.world_model(action_hist, token_hist)

        b, h, w, c = pred_logits.shape
        token_loss = self.token_loss_fn(pred_logits.view(b * h * w, c), target_tokens.view(b * h * w))
        reward_loss = self.reward_loss_fn(pred_reward, target_reward)
        done_loss = self.done_loss_fn(pred_done_logits, target_done)
        total_loss = token_loss + reward_loss + done_loss

        if is_train:
            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            return total_loss.item(), token_loss.item(), reward_loss.item(), done_loss.item(), grad_norm.item()
        else:
            return total_loss.item(), token_loss.item(), reward_loss.item(), done_loss.item()

    def _evaluate(self):
        """Runs a full evaluation on the validation set."""
        self.world_model.eval()
        total_loss, token_loss, reward_loss, done_loss = 0, 0, 0, 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                t_loss, tk_loss, r_loss, d_loss = self._run_batch(batch, is_train=False)
                total_loss += t_loss
                token_loss += tk_loss
                reward_loss += r_loss
                done_loss += d_loss
                num_batches += 1

        return {
            'total': total_loss / num_batches,
            'token': token_loss / num_batches,
            'reward': reward_loss / num_batches,
            'done': done_loss / num_batches,
        }

    def train(self, num_epochs):
        """Main training loop."""
        print("Starting Transformer world model training...")
        if isinstance(self.world_model, nn.DataParallel):
            self.world_model.module.token_embedding.weight.data.copy_(self.vq_vae_model.vq_layer.embeddings.data)
        else:
            self.world_model.token_embedding.weight.data.copy_(self.vq_vae_model.vq_layer.embeddings.data)
        print("Copied VQ-VAE weights to world model token embedding.")

        global_step = 0
        log_freq = self.config.get('log_freq', 10)
        val_freq = self.config.get('val_freq', 200)
        checkpoint_freq = self.config.get('checkpoint_freq', 5000)

        for epoch in range(1, num_epochs + 1):
            self.world_model.train()
            epoch_progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
            for batch in epoch_progress:
                global_step += 1
                total_loss, token_loss, reward_loss, done_loss, grad_norm = self._run_batch(batch, is_train=True)

                if self.logger:
                    self.logger.log_metrics({
                        'train/total_loss': total_loss, 'train/token_loss': token_loss,
                        'train/reward_loss': reward_loss, 'train/done_loss': done_loss,
                        'train/grad_norm': grad_norm, 'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=global_step)

                if global_step % log_freq == 0:
                    log_str = (
                        f"\n  +-----------------+----------+\n"
                        f"  |   Training      |  Value   |\n"
                        f"  +-----------------+----------+\n"
                        f"  | Step            | {global_step:<8} |\n"
                        f"  | Avg Total Loss  | {total_loss:<8.4f} |\n"
                        f"  | Avg Token Loss  | {token_loss:<8.4f} |\n"
                        f"  | Avg Reward Loss | {reward_loss:<8.4f} |\n"
                        f"  | Avg Done Loss   | {done_loss:<8.4f} |\n"
                        f"  | Grad Norm       | {grad_norm:<8.4f} |\n"
                        f"  +-----------------+----------+\n"
                    )
                    tqdm.write(log_str)

                if self.val_dataloader and global_step % val_freq == 0:
                    val_losses = self._evaluate()
                    if self.logger:
                        self.logger.log_metrics({f'val/{k}_loss': v for k, v in val_losses.items()}, step=global_step)

                    val_log_str = (
                        f"\n  +-----------------+----------+\n"
                        f"  |   Validation    |  Value   |\n"
                        f"  +-----------------+----------+\n"
                        f"  | Step            | {global_step:<8} |\n"
                        f"  | Avg Total Loss  | {val_losses['total']:<8.4f} |\n"
                        f"  | Avg Token Loss  | {val_losses['token']:<8.4f} |\n"
                        f"  | Avg Reward Loss | {val_losses['reward']:<8.4f} |\n"
                        f"  | Avg Done Loss   | {val_losses['done']:<8.4f} |\n"
                        f"  +-----------------+----------+\n"
                    )
                    tqdm.write(val_log_str)

                    self.world_model.train()  # Set back to training mode

                if global_step % checkpoint_freq == 0:
                    model_state = self.world_model.module.state_dict() if isinstance(self.world_model,
                                                                                     nn.DataParallel) else self.world_model.state_dict()
                    torch.save(model_state, TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_wm_step_{global_step}.pth")
                    tqdm.write(f"Saved model checkpoint at step {global_step}.")

        print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train History-Aware Transformer World Model")
    parser.add_argument("--config", type=str, default="default", help="Configuration name ('default', 'test').")
    parser.add_argument("--save-data-to", type=str, default=None, help="Path to save collected data.")
    parser.add_argument("--load-data-from", type=str, default=None, help="Path to load data, skipping collection.")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the logging run.")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to a model checkpoint to resume training.")
    args = parser.parse_args()

    config = get_config(args.config)
    print(f"Loaded configuration: '{args.config}'")

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        print(f"Could not set start method (possibly already set): {e}")

    if not VQ_VAE_CHECKPOINT_FILENAME.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: VQ-VAE Checkpoint {VQ_VAE_CHECKPOINT_FILENAME} not found.")

    data_buffer = None
    if args.load_data_from and Path(args.load_data_from).exists():
        print(f"Loading data from {args.load_data_from}...")
        data_buffer = torch.load(args.load_data_from, map_location='cpu')
    else:
        start_time = time.time()
        data_buffer = collect_data_parallel(
            num_steps=config["num_steps"], device=config["device"],
            num_workers=config["num_collection_workers"], env_name=config["env_name"],
            max_episode_steps=config["max_episode_steps_collect"]
        )
        print(f"Data collection took {time.time() - start_time:.2f} seconds.")
        if data_buffer and args.save_data_to:
            Path(args.save_data_to).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data_buffer, args.save_data_to)
            print(f"Data saved to {args.save_data_to}.")

    if not data_buffer:
        raise RuntimeError("ERROR: No data collected or loaded. Exiting.")

    full_dataset = TransformerHistoryDataset(data_buffer, config["history_length"])
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(config["validation_split"] * dataset_size))
    np.random.seed(config["random_seed"])
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        full_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=config["num_loader_workers"],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=config["num_loader_workers"],
        pin_memory=True if config['device'] == 'cuda' else False
    )

    world_model = WorldModelTransformer(
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        grid_size=config['grid_size'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['max_seq_len'],
        action_dim=config['action_dim'],
        codebook_size=VQVAE_NUM_EMBEDDINGS,
        vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
    ).to(
        config['device'])
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        print(f"Loading pre-trained model from {args.checkpoint_path}...")
        world_model.load_state_dict(torch.load(args.checkpoint_path, map_location=config['device']))

    vq_vae_model = VQVAE().to(config['device'])
    vq_vae_model.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=config['device']))
    vq_vae_model.eval()

    logger = ExperimentLogger(log_dir="logs/transformer_wm_logs", experiment_name="transformer_wm_training")
    run_name = args.run_name if args.run_name else f"{args.config}_{int(time.time())}"
    logger.start_run(run_name=run_name, config=config)

    trainer = WorldModelTransformerTrainer(
        world_model,
        vq_vae_model,
        config,
        train_loader,
        val_loader,
        logger
    )

    start_train_time = time.time()
    trainer.train(num_epochs=config["epochs"])
    print(f"Total training took {time.time() - start_train_time:.2f} seconds.")
    logger.end_run()

    TRANSFORMER_WM_CHECKPOINTS_DIR.mkdir(exist_ok=True)
    final_filename = TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_world_model_{args.config}.pth"
    torch.save(world_model.state_dict(), final_filename)
    print(f"Final model saved to {final_filename}")
