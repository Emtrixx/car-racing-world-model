import argparse
import multiprocessing as mp
import os
import random
import time

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from src.play_game_sb3 import SB3_MODEL_PATH
from src.utils import (
    ENV_NAME,
    ACTION_DIM,
    DEVICE, WM_CHECKPOINT_FILENAME_TRANSFORMER, VQ_VAE_CHECKPOINT_FILENAME, WorldModelDataCollector,
    make_env_sb3, NUM_STACK, TRANSFORMER_WM_CHECKPOINTS_DIR
)
from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM, VQVAE
from src.logger import ExperimentLogger
from src.transformer_world_model import WorldModelTransformer, TRANSFORMER_EMBED_DIM, TRANSFORMER_NUM_HEADS, \
    TRANSFORMER_NUM_LAYERS, TRANSFORMER_FF_DIM, TRANSFORMER_DROPOUT_RATE
from src.vq_conv_vae import GRID_SIZE
from src.utils import IMAGES_DIR

# --- Configuration ---
# Training Hyperparameters
NUM_STEPS = 1_000_000
WM_EPOCHS = 10
WM_BATCH_SIZE = 32
WM_LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 16
MAX_GRAD_NORM = 1.0

# Parallelism Configuration
NUM_COLLECTION_WORKERS = 4
NUM_LOADER_WORKERS = 4

# Environment settings for data collection
MAX_EPISODE_STEPS_COLLECT = 1000


def get_config(name="default"):
    configs = {
        "default": {
            "env_name": ENV_NAME,
            'action_dim': ACTION_DIM,
            'num_steps': NUM_STEPS,
            'epochs': WM_EPOCHS,
            'learning_rate': WM_LEARNING_RATE,
            'batch_size': WM_BATCH_SIZE,
            'sequence_length': SEQUENCE_LENGTH,
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
            #
            "max_seq_len": 256
        }
    }
    # test configuration for quick runs
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "num_steps": 500,
        "epochs": 3,
        "batch_size": 4,
        "sequence_length": 4,
        "num_collection_workers": 2,
        "num_loader_workers": 2,
        "max_episode_steps_collect": 100,
        "dropout_rate": 0.1,
    })
    return configs[name]


# --- Worker function for parallel data collection ---
def collect_sequences_worker(worker_id, num_steps_to_collect_by_worker, env_name_str,
                             device_str_for_worker,
                             max_episode_steps_collect_int):
    try:
        import os
        import sys
        sys.path.insert(0, os.path.abspath("src"))
        import time
        import torch
        import gymnasium as gym

        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Starting, assigned {num_steps_to_collect_by_worker} episodes. Device: {device_str_for_worker}")

        # --- Initialize Environment ---
        env = make_env_sb3(
            env_id=ENV_NAME,
            frame_stack_num=NUM_STACK,
            gamma=0.99,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps_collect_int,
        )

        print(f"[Worker {worker_id}] Observation space: {env.observation_space}")
        print(f"[Worker {worker_id}] Action space: {env.action_space}")

        # --- Load PPO Model ---
        print(f"Loading trained SB3 PPO agent from: {SB3_MODEL_PATH}")
        if not SB3_MODEL_PATH.exists():
            print(f"ERROR: SB3 PPO Model not found at {SB3_MODEL_PATH}")
            if hasattr(env, 'close'): env.close()
            return
        try:
            ppo_agent = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=env,
                                 deterministic=False)
            print(f"Successfully loaded SB3 PPO agent. Agent device: {ppo_agent.device}")
        except Exception as e:
            print(f"ERROR loading SB3 PPO agent: {e}")
            if hasattr(env, 'close'): env.close()
            import traceback
            traceback.print_exc()
            return

        # --- Load VQ-VAE Model ---
        vq_vae = VQVAE().to(DEVICE)

        try:
            print(f"Loading trained model from: {VQ_VAE_CHECKPOINT_FILENAME}")
            vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        except FileNotFoundError:
            print("VQ-VAE Model file not found")
            return

        # --- Prepare for Data Collection ---
        collector = WorldModelDataCollector(env, ppo_agent, vq_vae, device_str_for_worker)
        collector.collect_steps(num_steps=num_steps_to_collect_by_worker)

        env.close()
        print(
            f"[Worker {worker_id}, PID {os.getpid()}] Finished data collection. Collected {len(collector.replay_buffer)} sequences.")

        # --- Prepare Data for Saving ---
        actions = torch.stack([s['action'] for s in collector.replay_buffer])
        rewards = torch.stack([s['reward'] for s in collector.replay_buffer])
        dones = torch.stack([s['done'] for s in collector.replay_buffer])
        next_tokens = torch.stack([s['next_tokens'] for s in collector.replay_buffer])

        data_to_save = {
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'next_tokens': next_tokens
        }
        temp_data_dir = "./tmp_worker_data"
        if not os.path.exists(temp_data_dir):
            try:
                os.makedirs(temp_data_dir)
                print(f"[Worker {worker_id}] Created directory: {temp_data_dir}")
            except OSError as e:
                print(f"[Worker {worker_id}] Error creating directory {temp_data_dir}: {e}")
                temp_data_dir = ".."

        timestamp = int(time.time() * 1000)
        filename = os.path.join(temp_data_dir, f"temp_worker_data_{worker_id}_{timestamp}.pt")

        try:
            print(f"[Worker {worker_id}] Attempting to save data to {filename}...")
            torch.save(data_to_save, filename)
            print(f"[Worker {worker_id}] Successfully saved data to {filename}.")
            return filename
        except Exception as e_save:
            print(f"[Worker {worker_id}] ERROR saving data to {filename}: {e_save}")
            import traceback
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"[Worker {worker_id}, PID {os.getpid()}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        if 'worker_env' in locals():
            env.close()
        return None


def collect_sequences_for_transformer(num_steps_total, device_str_main,
                                      num_collection_workers_int,
                                      env_name_str_for_worker,
                                      max_episode_steps_collect_int
                                      ):
    print(
        f"Starting parallel collection with {num_collection_workers_int} workers for {num_steps_total} total episodes...")

    if num_collection_workers_int <= 0:
        print("Warning: num_collection_workers is not positive. Defaulting to 1 worker (serial collection).")
        num_collection_workers_int = 1

    steps_per_worker = [num_steps_total // num_collection_workers_int] * num_collection_workers_int
    remainder_episodes = num_steps_total % num_collection_workers_int
    for i in range(remainder_episodes):
        steps_per_worker[i] += 1

    worker_args_list = []
    actual_workers_to_spawn = 0
    for worker_id in range(num_collection_workers_int):
        if steps_per_worker[worker_id] == 0:
            print(f"Skipping worker {worker_id} as it has no episodes assigned.")
            continue
        actual_workers_to_spawn += 1
        args = (
            worker_id,
            steps_per_worker[worker_id],
            env_name_str_for_worker,
            device_str_main,
            max_episode_steps_collect_int
        )
        worker_args_list.append(args)

    if not worker_args_list:
        print("No episodes to collect or no workers to assign after distribution. Skipping parallel collection.")
        return []

    pool_size = min(actual_workers_to_spawn, num_collection_workers_int)
    if pool_size == 0:
        print("No workers to spawn. Returning empty list.")
        return []

    print(f"Distributing work: {steps_per_worker} episodes per worker. Spawning {pool_size} worker processes.")

    with mp.Pool(processes=pool_size) as pool:
        filepath_results = pool.starmap(collect_sequences_worker, worker_args_list)

    all_worker_data_dicts = []
    for worker_idx, filepath_result in enumerate(filepath_results):
        worker_actual_id = worker_args_list[worker_idx][0]
        if filepath_result and os.path.exists(filepath_result):
            try:
                print(f"Loading data from worker {worker_actual_id}'s file: {filepath_result}")
                worker_data = torch.load(filepath_result, weights_only=False)
                all_worker_data_dicts.append(worker_data)
                print(
                    f"Successfully loaded {len(worker_data)} sequences from worker {worker_actual_id} (file: {filepath_result}).")
                try:
                    os.remove(filepath_result)
                    print(f"Removed temporary file: {filepath_result}")
                except OSError as e_remove:
                    print(f"Warning: Could not remove temporary file {filepath_result}: {e_remove}")
            except Exception as e_load:
                print(f"ERROR loading data from worker {worker_actual_id}'s file {filepath_result}: {e_load}")
        elif filepath_result:
            print(f"Worker {worker_actual_id} returned filepath {filepath_result}, but file not found.")
        else:
            print(f"Worker {worker_actual_id} failed to produce a data file.")

    if not all_worker_data_dicts:
        print("No valid data loaded from any worker.")
        return None

    final_data_buffer = {
        'actions': torch.cat([d['actions'] for d in all_worker_data_dicts], dim=0),
        'rewards': torch.cat([d['rewards'] for d in all_worker_data_dicts], dim=0),
        'dones': torch.cat([d['dones'] for d in all_worker_data_dicts], dim=0),
        'next_tokens': torch.cat([d['next_tokens'] for d in all_worker_data_dicts], dim=0),
    }

    print(f"Finished collecting. Total transitions from all workers: {len(final_data_buffer['actions'])}.")
    temp_data_dir = "./tmp_worker_data"
    if os.path.exists(temp_data_dir) and not os.listdir(temp_data_dir):
        try:
            os.rmdir(temp_data_dir)
            print(f"Removed empty temporary directory: {temp_data_dir}")
        except OSError as e_rmdir:
            print(f"Warning: Could not remove temporary directory {temp_data_dir}: {e_rmdir}")
    elif os.path.exists(temp_data_dir) and os.listdir(temp_data_dir):
        print(f"Warning: Temporary directory {temp_data_dir} is not empty. Manual cleanup might be needed.")

    return final_data_buffer


class TransformerSequenceDataset(Dataset):
    def __init__(self, data_dict, sequence_length):
        self.data = data_dict
        self.sequence_length = sequence_length
        # We cannot start at index 0 because we need the tokens from t-1.
        # Valid start indices are [1, ..., N - sequence_length].
        # So the number of sequences is (N - L) - 1 + 1 = N - L.
        self.num_sequences = len(data_dict['actions']) - sequence_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Sampler gives idx from 0 to N-L-1. We map it to start from 1 to N-L.
        start = idx + 1
        end = start + self.sequence_length

        prev_tokens_start = start - 1
        prev_tokens_end = end - 1

        return {
            'actions': self.data['actions'][start:end],
            'rewards': self.data['rewards'][start:end],
            'dones': self.data['dones'][start:end],
            'next_tokens': self.data['next_tokens'][start:end],
            'prev_tokens': self.data['next_tokens'][prev_tokens_start:prev_tokens_end]
        }


class WorldModelTransformerTrainer:
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

    def _evaluate(self):
        self.world_model.eval()
        total_val_token_loss, total_val_reward_loss, total_val_done_loss = 0, 0, 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                actions = batch['actions']
                prev_tokens = batch['prev_tokens']
                next_tokens_gt = batch['next_tokens']
                rewards_gt = batch['rewards']
                dones_gt = batch['dones']

                pred_logits, pred_reward, pred_done_logits, _, _ = self.world_model(actions, prev_tokens)

                b, s, h, w, c = pred_logits.shape
                token_loss = self.token_loss_fn(
                    pred_logits.reshape(b * s * h * w, c),
                    next_tokens_gt.reshape(b * s * h * w)
                )
                reward_loss = self.reward_loss_fn(pred_reward, rewards_gt)
                done_loss = self.done_loss_fn(pred_done_logits, dones_gt)

                total_val_token_loss += token_loss.item()
                total_val_reward_loss += reward_loss.item()
                total_val_done_loss += done_loss.item()
                num_val_batches += 1

        avg_val_token_loss = total_val_token_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_reward_loss = total_val_reward_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_done_loss = total_val_done_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_total_loss = avg_val_token_loss + avg_val_reward_loss + avg_val_done_loss

        return {
            'total': avg_val_total_loss,
            'token': avg_val_token_loss,
            'reward': avg_val_reward_loss,
            'done': avg_val_done_loss,
        }

    def train(self, num_epochs):
        print("Starting Transformer world model training...")
        if isinstance(self.world_model, nn.DataParallel):
            self.world_model.module.token_embedding.weight.data.copy_(
                self.vq_vae_model.vq_layer.embedding.weight.data
            )
        else:
            self.world_model.token_embedding.weight.data.copy_(
                self.vq_vae_model.vq_layer.embeddings.data
            )
        print("Copied VQ-VAE weights to world model token embedding.")

        self.world_model.train()

        global_step = 0
        log_freq = self.config.get('log_freq', 10)
        val_freq = self.config.get('val_freq', 200)
        checkpoint_freq = self.config.get('checkpoint_freq', 5000)

        # Accumulators for logging averages
        running_total_loss, running_token_loss, running_reward_loss, running_done_loss, running_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

        for epoch in range(1, num_epochs + 1):
            epoch_progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
            for batch in epoch_progress:
                global_step += 1

                for key in batch:
                    batch[key] = batch[key].to(self.device)

                actions = batch['actions']
                prev_tokens = batch['prev_tokens']
                next_tokens_gt = batch['next_tokens']
                rewards_gt = batch['rewards']
                dones_gt = batch['dones']

                pred_logits, pred_reward, pred_done_logits, _, _ = self.world_model(actions, prev_tokens)

                b, s, h, w, c = pred_logits.shape
                token_loss = self.token_loss_fn(
                    pred_logits.reshape(b * s * h * w, c),
                    next_tokens_gt.reshape(b * s * h * w)
                )
                reward_loss = self.reward_loss_fn(pred_reward, rewards_gt)
                done_loss = self.done_loss_fn(pred_done_logits, dones_gt)

                total_loss = token_loss + reward_loss + done_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()

                # Log training metrics
                if self.logger:
                    self.logger.log_metrics({
                        'train_total_loss': total_loss.item(),
                        'train_token_loss': token_loss.item(),
                        'train_reward_loss': reward_loss.item(),
                        'train_done_loss': done_loss.item(),
                        'grad_norm': grad_norm.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }, step=global_step)

                # Add to running totals for logging
                running_total_loss += total_loss.item()
                running_token_loss += token_loss.item()
                running_reward_loss += reward_loss.item()
                running_done_loss += done_loss.item()
                running_grad_norm += grad_norm.item()

                if global_step % log_freq == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    avg_total_loss = running_total_loss / log_freq
                    avg_token_loss = running_token_loss / log_freq
                    avg_reward_loss = running_reward_loss / log_freq
                    avg_done_loss = running_done_loss / log_freq
                    avg_grad_norm = running_grad_norm / log_freq

                    log_str = (
                        f"\n  +-----------------+----------+\n"
                        f"  |    Training     |  Value   |\n"
                        f"  +-----------------+----------+\n"
                        f"  | Step            | {global_step:<8} |\n"
                        f"  | Avg Total Loss  | {avg_total_loss:<8.4f} |\n"
                        f"  | Avg Token Loss  | {avg_token_loss:<8.4f} |\n"
                        f"  | Avg Reward Loss | {avg_reward_loss:<8.4f} |\n"
                        f"  | Avg Done Loss   | {avg_done_loss:<8.4f} |\n"
                        f"  | Avg Grad Norm   | {avg_grad_norm:<8.4f} |\n"
                        f"  | Learning Rate   | {lr:<8.6f} |\n"
                        f"  +-----------------+----------+"
                    )
                    tqdm.write(log_str)

                    # Reset accumulators
                    running_total_loss, running_token_loss, running_reward_loss, running_done_loss, running_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

                if self.val_dataloader and global_step > 0 and global_step % val_freq == 0:
                    val_losses = self._evaluate()
                    if self.logger:
                        self.logger.log_metrics({
                            f'val_total_loss': val_losses['total'],
                            f'val_token_loss': val_losses['token'],
                            f'val_reward_loss': val_losses['reward'],
                            f'val_done_loss': val_losses['done'],
                        }, step=global_step)

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
                    self.world_model.train()

                if global_step > 0 and global_step % checkpoint_freq == 0:
                    model_state_to_save = self.world_model.module.state_dict() if isinstance(self.world_model,
                                                                                             nn.DataParallel) else self.world_model.state_dict()
                    torch.save(model_state_to_save,
                               TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_world_model_step_{global_step}.pth")
                    tqdm.write(f"Saved model checkpoint at step {global_step}.")
            epoch_progress.close()

        print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer World Model")
    parser.add_argument("--config", type=str, default="default",
                        help="Name of the configuration to use (e.g., 'default', 'test').")
    parser.add_argument("--save-data-to", type=str, default=None,
                        help="Path to save the collected data to. Data is not saved unless this is specified.")
    parser.add_argument("--load-data-from", type=str, default=None,
                        help="Path to load data from, skipping collection.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name of the run for logging purposes.")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to a pre-trained model checkpoint to load before training. If provided, "
                             "the model will be loaded and training will continue from that point.")
    args = parser.parse_args()

    config = get_config(args.config)
    print(f"Loaded configuration: '{args.config}'")

    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        print(f"Could not set start method (possibly already set or not allowed): {e}")
        pass

    print(f"Starting Transformer World Model training on device: {config['device']}")

    if not os.path.exists(VQ_VAE_CHECKPOINT_FILENAME):
        print(
            f"CRITICAL ERROR: VAE Checkpoint {VQ_VAE_CHECKPOINT_FILENAME} not found. Exiting.")
        exit()

    sequence_data_buffer = None
    if args.load_data_from:
        if os.path.exists(args.load_data_from):
            print(f"Loading sequence data from {args.load_data_from}...")
            sequence_data_buffer = torch.load(args.load_data_from, map_location=config['device'])
            print("Data loaded successfully.")
        else:
            print(f"ERROR: Data file not found at {args.load_data_from}. Exiting.")
            exit()
    else:
        # This block is for data collection
        start_collect_time = time.time()
        sequence_data_buffer = collect_sequences_for_transformer(
            num_steps_total=config["num_steps"],
            device_str_main=config["device"],
            num_collection_workers_int=config["num_collection_workers"],
            env_name_str_for_worker=config["env_name"],
            max_episode_steps_collect_int=config["max_episode_steps_collect"]
        )
        print(f"Sequence data collection took {time.time() - start_collect_time:.2f} seconds.")

        if sequence_data_buffer and args.save_data_to:
            print(f"Saving collected data to {args.save_data_to}...")
            save_dir = os.path.dirname(args.save_data_to)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(sequence_data_buffer, args.save_data_to)
            print(f"Data saved to {args.save_data_to}.")

    if not sequence_data_buffer:
        print("ERROR: No sequence data collected or loaded. Exiting.")
        exit()

    print("Splitting data into training and validation sets...")
    full_dataset = TransformerSequenceDataset(sequence_data_buffer, config["sequence_length"])

    validation_split_ratio = config.get("validation_split", 0.1)
    shuffle_dataset = True
    random_seed_split = config.get("random_seed", 42)
    print(f"Random seed: {random_seed_split}")

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(validation_split_ratio * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed_split)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    print(f"Total sequences: {dataset_size}")
    print(f"Training sequences: {len(train_indices)}")
    print(f"Validation sequences: {len(val_indices)}")

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(
        full_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=config["num_loader_workers"],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    val_dataloader = DataLoader(
        full_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=config["num_loader_workers"],
        pin_memory=True if config['device'] == 'cuda' else False
    )

    world_model_transformer = WorldModelTransformer(
        vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
        codebook_size=VQVAE_NUM_EMBEDDINGS,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        grid_size=config['grid_size'],
        dropout_rate=config['dropout_rate'],
        max_seq_len=config['max_seq_len']
    )
    checkpoint_path = args.checkpoint_path
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading pre-trained Transformer World Model from {checkpoint_path}...")
        try:
            world_model_transformer.load_state_dict(torch.load(checkpoint_path, map_location=config['device']))
            print("Pre-trained model loaded successfully.")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            import traceback

            traceback.print_exc()

    world_model_transformer.to(config['device'])

    vq_vae_model = VQVAE(embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS)
    vq_vae_model.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=config['device']))
    vq_vae_model.to(config['device'])
    vq_vae_model.eval()

    # if torch.cuda.device_count() > 1:
    #     print(f"Using nn.DataParallel for Transformer model training across {torch.cuda.device_count()} GPUs.")
    #     world_model_transformer = nn.DataParallel(world_model_transformer)

    logger = ExperimentLogger(log_dir="logs/transformer_wm_logs", experiment_name="transformer_wm_training")
    run_name = args.run_name if args.run_name else f"{args.config}_{int(time.time())}"
    logger.start_run(run_name=run_name, config=config)

    trainer = WorldModelTransformerTrainer(
        world_model_transformer,
        vq_vae_model,
        config,
        train_dataloader,
        val_dataloader,
        logger
    )

    print("Starting Transformer World Model training via WorldModelTransformerTrainer...")
    start_train_time = time.time()
    trainer.train(num_epochs=config["epochs"])
    print(f"Transformer World Model training took {time.time() - start_train_time:.2f} seconds.")

    logger.end_run()

    try:
        model_state_to_save = world_model_transformer.module.state_dict() if isinstance(world_model_transformer,
                                                                                        nn.DataParallel) else world_model_transformer.state_dict()
        filename = f"transformer_world_model_{args.config}.pth"
        torch.save(model_state_to_save, TRANSFORMER_WM_CHECKPOINTS_DIR / filename)
        print(f"Transformer World Model saved to {filename}")
    except Exception as e:
        print(f"Error saving Transformer World Model: {e}")
