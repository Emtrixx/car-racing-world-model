import argparse
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from src.logger import ExperimentLogger
from src.transformer_world_model import WorldModelTransformer, TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS, \
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT_RATE
from src.utils import (
    ENV_NAME,
    ACTION_DIM,
    DEVICE, VQ_VAE_CHECKPOINT_FILENAME,
    TRANSFORMER_WM_CHECKPOINTS_DIR
)
from src.vq_conv_vae import GRID_SIZE
from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM, VQVAE
from src.utils_wm import collect_data_for_transformer
from src.utils import LOGS_DIR

# --- Configuration ---
# Training Hyperparameters
NUM_STEPS = 1_000_000
WM_EPOCHS = 10
WM_BATCH_SIZE = 64
WM_LEARNING_RATE = 1e-4
HISTORY_LENGTH = 32  # Renamed from SEQUENCE_LENGTH
MAX_GRAD_NORM = 1.0

# Parallelism Configuration
NUM_COLLECTION_WORKERS = 4
NUM_LOADER_WORKERS = 12

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
            # At least the length of the history multiplied by (tokens per state + 1 for action).
            "max_seq_len": 4096  # for positional encoding
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
    configs["profile"] = configs["default"].copy()
    configs["profile"].update({
        "num_steps": 12000,
        "epochs": 1,
    })
    return configs[name]


class TransformerHistoryDataset(Dataset):
    """
    Dataset to provide history of states and actions for the world model.
    This version is aware of episode boundaries to avoid creating sequences
    that cross from one episode to another.
    """

    def __init__(self, data_dict, history_length):
        self.data = data_dict
        self.history_length = history_length
        self.total_steps = len(data_dict['actions'])

        # is_first_steps is a boolean tensor indicating the start of an episode.
        is_first_steps = data_dict.get('is_first_steps', torch.zeros(self.total_steps, dtype=torch.bool))
        if is_first_steps.ndim > 1:
            is_first_steps = is_first_steps.squeeze(1)

        self.valid_indices = []
        for i in range(self.total_steps - self.history_length):
            # A sequence is valid if it does not contain any episode starts
            # after the first element. The first element (i) can be a start.
            window = is_first_steps[i + 1: i + self.history_length + 1]
            if not torch.any(window):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map the requested index to a valid starting index in the dataset
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.history_length

        # History of actions and previous states
        action_history = self.data['actions'][start_idx:end_idx]
        token_history = self.data['prev_tokens'][start_idx:end_idx]

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

    def train(self, num_epochs, copy_vq_weights=True, profile=False):
        """Main training loop."""
        print("Starting Transformer world model training...")
        if copy_vq_weights:
            if isinstance(self.world_model, nn.DataParallel):
                self.world_model.module.token_embedding.weight.data.copy_(self.vq_vae_model.vq_layer.embeddings.data)
            else:
                self.world_model.token_embedding.weight.data.copy_(self.vq_vae_model.vq_layer.embeddings.data)
            print("Copied VQ-VAE weights to world model token embedding.")

        global_step = 0
        log_freq = self.config.get('log_freq', 10)
        val_freq = self.config.get('val_freq', 200)
        checkpoint_freq = self.config.get('checkpoint_freq', 5000)

        profiler = None
        if profile:
            print("Profiling enabled. The profiler will start after a few warmup steps.")
            log_dir = self.logger.log_dir if self.logger else LOGS_DIR / "profiler_logs"
            run_name = self.logger.experiment_name if self.logger and self.logger.experiment_name else f"run_{int(time.time())}"
            trace_dir = Path(log_dir) / run_name / "profile"
            trace_dir.mkdir(parents=True, exist_ok=True)
            print(f"Profiler traces will be saved to: {trace_dir}")

            activities = [torch.profiler.ProfilerActivity.CPU]
            if 'cuda' in str(self.device):
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=20, warmup=5, active=10, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )
            profiler.start()

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

                if profiler:
                    profiler.step()

        if profiler:
            profiler.stop()
            print("Profiling finished. You can view the trace with TensorBoard.")

        print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train History-Aware Transformer World Model")
    parser.add_argument("--config", type=str, default="default", help="Configuration name ('default', 'test').")
    parser.add_argument("--save-data-to", type=str, default=None, help="Path to save collected data.")
    parser.add_argument("--load-data-from", type=str, default=None, help="Path to load data, skipping collection.")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the logging run.")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to a model checkpoint to resume training.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")
    args = parser.parse_args()

    config = get_config(args.config)
    print(f"Loaded configuration: '{args.config}'")
    print(f"Device: {config['device']}")

    if "cuda" in str(config['device']):
        torch.set_float32_matmul_precision('high')

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
        limit = config["num_steps"]
        if limit < len(data_buffer['actions']):
            # Limit the dataset to the first 'limit' sequences
            for key in data_buffer:
                data_buffer[key] = data_buffer[key][:limit]
    else:
        start_time = time.time()
        data_buffer = collect_data_for_transformer(
            num_steps=config["num_steps"],
            device=config["device"],
            num_workers=config["num_collection_workers"],
            env_name=config["env_name"],
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
        pin_memory=True if "cuda" in str(config['device']) else False
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=config["num_loader_workers"],
        pin_memory=True if "cuda" in str(config['device']) else False
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

    # Compile the model for performance
    world_model = torch.compile(world_model)

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
    trainer.train(num_epochs=config["epochs"], copy_vq_weights=args.checkpoint_path is None, profile=args.profile)
    print(f"Total training took {time.time() - start_train_time:.2f} seconds.")
    logger.end_run()

    TRANSFORMER_WM_CHECKPOINTS_DIR.mkdir(exist_ok=True)
    final_filename = TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_world_model_{args.config}.pth"
    torch.save(world_model.state_dict(), final_filename)
    print(f"Final model saved to {final_filename}")
