import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from src.utils import (
    ENV_NAME,  # Default: "CarRacing-v3"
    ACTION_DIM,  # Default: 3
    DEVICE, WM_CHECKPOINT_FILENAME_GRU, VQ_VAE_CHECKPOINT_FILENAME
)
from src.utils_wm import collect_sequences_for_gru
from src.vq_conv_vae import VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM, VQVAE
from src.world_model import GRU_HIDDEN_DIM, GRU_NUM_LAYERS, WorldModelGRU
from src.utils import GRU_WM_CHECKPOINTS_DIR
from src.utils import LOGS_DIR

# --- Configuration ---
# Training Hyperparameters
NUM_STEPS = 1_000_000  # Number of steps to collect for training the world model
WM_EPOCHS = 10  # Number of epochs to train the world model
WM_BATCH_SIZE = 128  # Sequences per batch
WM_LEARNING_RATE = 1e-4  # Learning rate for world model optimizer
SEQUENCE_LENGTH = 256  # Length of sequences to train on
MAX_GRAD_NORM = 1.0  # Max gradient norm for clipping

# Parallelism Configuration
NUM_COLLECTION_WORKERS = 4  # For multiprocessing data collection
NUM_LOADER_WORKERS = 12  # For DataLoader for PyTorch training

# Environment settings for data collection
MAX_EPISODE_STEPS_COLLECT = 1000  # Max steps per episode in the collection environment


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
            "gru_hidden_dim": GRU_HIDDEN_DIM,  # GRU Hidden Dimension per layer
            "num_gru_layers": GRU_NUM_LAYERS,  # Number of GRU layers
            "dropout_rate": 0.1,  # Dropout rate
        }
    }
    # test configuration for quick runs
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "num_steps": 1000,
        "epochs": 3,
        "batch_size": 4,
        "sequence_length": 4,
        "num_collection_workers": 2,
        "num_loader_workers": 2,
        "max_episode_steps_collect": 100,
        # "gru_hidden_dim": 32,
        # "num_gru_layers": 2,
        "dropout_rate": 0.1,
    })
    configs["profile"] = configs["default"].copy()
    configs["profile"].update({
        "num_steps": 5000,
        "epochs": 1,
    })
    return configs[name]


class SequenceDataset(Dataset):
    """
    A PyTorch Dataset for handling the structured dictionary of sequence data.
    This dataset returns entire sequences of a fixed length.
    """

    def __init__(self, data_dict, sequence_length):
        """
        Args:
            data_dict (dict): A dictionary where keys are 'actions', 'rewards', etc.,
                              and values are tensors of the entire dataset.
            sequence_length (int): The length of the sequences to return.
        """
        self.data = data_dict
        self.sequence_length = sequence_length
        # The number of possible start points for a sequence
        self.num_sequences = len(data_dict['actions']) - sequence_length + 1

    def __len__(self):
        """Returns the total number of possible sequences."""
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns a dictionary containing one sequence of data starting at the given index.
        """
        # Define the start and end of the sequence slice
        start = idx
        end = idx + self.sequence_length

        # Slice each tensor to get the data for the full sequence
        return {
            'actions': self.data['actions'][start:end],
            'rewards': self.data['rewards'][start:end],
            'dones': self.data['dones'][start:end],
            'next_tokens': self.data['next_tokens'][start:end]
        }


# --- Main Execution ---
class GruWorldModelTrainer:
    """Handles the training loop for the WorldModelGRU."""

    def __init__(self, world_model, vq_vae_model, config, train_dataloader, val_dataloader=None):
        self.world_model = world_model
        self.vq_vae_model = vq_vae_model  # Needed for weight copying
        self.config = config
        self.device = config['device']
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Define optimizers and loss functions
        self.optimizer = Adam(self.world_model.parameters(), lr=config['learning_rate'])
        self.token_loss_fn = nn.CrossEntropyLoss()
        self.reward_loss_fn = nn.MSELoss()
        self.done_loss_fn = nn.BCEWithLogitsLoss()

        # For logging and plotting
        self.loss_history = {
            'train_total': [], 'train_token': [], 'train_reward': [], 'train_done': [],
            'val_total': [], 'val_token': [], 'val_reward': [], 'val_done': []
        }
        self.steps_history = []  # For x-axis of training plots
        self.val_steps_history = []  # For x-axis of validation plots

    def plot_losses(self):
        """Plots the collected loss history and saves it to a file."""
        print("Plotting training and validation losses...")
        plt.figure(figsize=(12, 8))

        if self.loss_history['train_total']:
            plt.plot(self.steps_history, self.loss_history['train_total'], label='Train Total Loss')
            # plt.plot(self.steps_history, self.loss_history['train_token'], label='Train Token Loss', linestyle='--')
            # plt.plot(self.steps_history, self.loss_history['train_reward'], label='Train Reward Loss', linestyle='--')
            # plt.plot(self.steps_history, self.loss_history['train_done'], label='Train Done Loss', linestyle='--')

        if self.loss_history['val_total']:
            plt.plot(self.val_steps_history, self.loss_history['val_total'], label='Validation Total Loss',
                     linestyle=':')
            # plt.plot(self.val_steps_history, self.loss_history['val_token'], label='Val Token Loss', linestyle='-.')
            # plt.plot(self.val_steps_history, self.loss_history['val_reward'], label='Val Reward Loss', linestyle='-.')
            # plt.plot(self.val_steps_history, self.loss_history['val_done'], label='Val Done Loss', linestyle='-.')

        plt.title(f"World Model Training & Validation Loss (SeqLen {self.config['sequence_length']})")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        if self.loss_history['train_total'] or self.loss_history['val_total']:  # only add legend if there's data
            plt.legend()
        plt.grid(True)

        # Ensure the save directory exists
        save_dir = self.config.get("plot_save_dir", "images")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "world_model_loss_plot_with_val.png")  # New filename
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
        plt.close()

    def _evaluate(self):
        self.world_model.eval()
        total_val_token_loss, total_val_reward_loss, total_val_done_loss = 0, 0, 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                batch_size = batch['actions'].size(0)
                # Correctly get initial hidden state, handling DataParallel
                current_model_module = self.world_model.module if isinstance(self.world_model,
                                                                             nn.DataParallel) else self.world_model

                # prev_model_state is the recurrent state (e.g. GRU hidden) or None for Transformer
                prev_model_state = current_model_module.get_initial_hidden_state(batch_size, self.device)

                seq_token_loss, seq_reward_loss, seq_done_loss = 0, 0, 0
                sequence_length = batch['actions'].size(1)

                for t in range(sequence_length):
                    action_t = batch['actions'][:, t]
                    ground_truth_tokens_t = batch['next_tokens'][:, t]  # These are for the frame AFTER action_t
                    ground_truth_reward_t = batch['rewards'][:, t]
                    ground_truth_done_t = batch['dones'][:, t]

                    # For Transformer, block_tf_ratio and block_size are part of its own config,
                    # not passed dynamically by the trainer loop here.
                    pred_logits, pred_reward, pred_done_logits, next_model_state = self.world_model(
                        action_t, prev_model_state, ground_truth_tokens=ground_truth_tokens_t
                    )

                    b, h, w, c = pred_logits.shape
                    token_loss = self.token_loss_fn(
                        pred_logits.reshape(b * h * w, c),
                        ground_truth_tokens_t.reshape(b * h * w))
                    reward_loss = self.reward_loss_fn(pred_reward, ground_truth_reward_t)
                    done_loss = self.done_loss_fn(pred_done_logits, ground_truth_done_t)

                    seq_token_loss += token_loss
                    seq_reward_loss += reward_loss
                    seq_done_loss += done_loss
                    prev_model_state = next_model_state  # Update recurrent state for next step in sequence

                total_val_token_loss += (seq_token_loss / sequence_length)
                total_val_reward_loss += (seq_reward_loss / sequence_length)
                total_val_done_loss += (seq_done_loss / sequence_length)
                num_val_batches += 1

        avg_val_token_loss = total_val_token_loss / num_val_batches if num_val_batches > 0 else torch.tensor(0.0).to(
            self.device)
        avg_val_reward_loss = total_val_reward_loss / num_val_batches if num_val_batches > 0 else torch.tensor(0.0).to(
            self.device)
        avg_val_done_loss = total_val_done_loss / num_val_batches if num_val_batches > 0 else torch.tensor(0.0).to(
            self.device)
        avg_val_total_loss = avg_val_token_loss + avg_val_reward_loss + avg_val_done_loss

        return {
            'total': avg_val_total_loss.item(),
            'token': avg_val_token_loss.item(),
            'reward': avg_val_reward_loss.item(),
            'done': avg_val_done_loss.item(),
        }

    def train(self, num_epochs, copy_vq_weights=True, profile=False):
        """Main training loop that iterates over a DataLoader."""
        print("Starting world model training...")
        if copy_vq_weights:
            if isinstance(self.world_model, nn.DataParallel):
                self.world_model.module.token_embedding.weight.data.copy_(
                    self.vq_vae_model.vq_layer.embeddings.data
                )
            else:
                self.world_model.token_embedding.weight.data.copy_(
                    self.vq_vae_model.vq_layer.embeddings.data
                )
        print("Copied VQ-VAE weights to world model token embedding.")

        self.world_model.train()

        # Initialize step counters
        global_step = 0
        total_train_steps = len(self.train_dataloader) * num_epochs  # Use self.train_dataloader
        log_freq = self.config.get('log_freq', 100)
        val_freq = self.config.get('val_freq', log_freq * 5)  # val_freq from config
        checkpoint_freq = self.config.get('checkpoint_freq', 5000)

        profiler = None
        if profile:
            print("Profiling enabled. The profiler will start after a few warmup steps.")
            log_dir = LOGS_DIR / "gru_wm_logs"
            run_name = f"run_{int(time.time())}"
            trace_dir = Path(log_dir) / run_name / "profile"
            trace_dir.mkdir(parents=True, exist_ok=True)
            print(f"Profiler traces will be saved to: {trace_dir}")

            activities = [torch.profiler.ProfilerActivity.CPU]
            if 'cuda' in str(self.device):
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=50, warmup=5, active=10, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )
            profiler.start()

        for epoch in range(1, num_epochs + 1):
            for batch_idx, batch in enumerate(self.train_dataloader):  # Use self.train_dataloader
                global_step += 1

                for key in batch:
                    batch[key] = batch[key].to(self.device)

                # Initialize hidden state for the start of the sequences
                batch_size = batch['actions'].size(0)
                current_model_module = self.world_model.module if isinstance(self.world_model,
                                                                             nn.DataParallel) else self.world_model

                prev_model_state = current_model_module.get_initial_hidden_state(batch_size, self.device)

                total_token_loss, total_reward_loss, total_done_loss = 0, 0, 0
                sequence_length = batch['actions'].size(1)

                for t in range(sequence_length):
                    action_t = batch['actions'][:, t]
                    ground_truth_tokens_t = batch['next_tokens'][:, t]  # Tokens for frame AFTER action_t
                    ground_truth_reward_t = batch['rewards'][:, t]
                    ground_truth_done_t = batch['dones'][:, t]

                    # For Transformer, block_tf_ratio and block_size are part of its own config.
                    pred_logits, pred_reward, pred_done_logits, next_model_state = self.world_model(
                        action_t, prev_model_state, ground_truth_tokens=ground_truth_tokens_t
                    )

                    b, h, w, c = pred_logits.shape
                    token_loss = self.token_loss_fn(
                        pred_logits.reshape(b * h * w, c),
                        ground_truth_tokens_t.reshape(b * h * w))
                    reward_loss = self.reward_loss_fn(pred_reward, ground_truth_reward_t)
                    done_loss = self.done_loss_fn(pred_done_logits, ground_truth_done_t)

                    total_token_loss += token_loss
                    total_reward_loss += reward_loss
                    total_done_loss += done_loss
                    prev_model_state = next_model_state  # Update recurrent state

                avg_token_loss = total_token_loss / sequence_length
                avg_reward_loss = total_reward_loss / sequence_length
                avg_done_loss = total_done_loss / sequence_length
                total_loss = avg_token_loss + avg_reward_loss + avg_done_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()

                # Logging
                if global_step % log_freq == 0:
                    print(f"Epoch {epoch}/{num_epochs} | Step {global_step}/{total_train_steps} | "
                          f"Train Total Loss: {total_loss.item():.4f} | "  # Clarified Train
                          f"Train Token Loss: {avg_token_loss.item():.4f} | "
                          f"Train Reward Loss: {avg_reward_loss.item():.4f} | "
                          f"Train Done Loss: {avg_done_loss.item():.4f}")

                    # Store loss values for plotting
                    self.loss_history['train_total'].append(total_loss.item())
                    self.loss_history['train_token'].append(avg_token_loss.item())
                    self.loss_history['train_reward'].append(avg_reward_loss.item())
                    self.loss_history['train_done'].append(avg_done_loss.item())
                    self.steps_history.append(global_step)  # Record step for training loss

                # Validation step
                if self.val_dataloader and global_step > 0 and global_step % val_freq == 0:
                    val_losses = self._evaluate()
                    self.loss_history['val_total'].append(val_losses['total'])
                    self.loss_history['val_token'].append(val_losses['token'])
                    self.loss_history['val_reward'].append(val_losses['reward'])
                    self.loss_history['val_done'].append(val_losses['done'])
                    self.val_steps_history.append(global_step)  # Record step for validation loss

                    print(f"Epoch {epoch}/{num_epochs} | Step {global_step}/{total_train_steps} | "
                          f"Val Total Loss: {val_losses['total']:.4f} | Val Token: {val_losses['token']:.4f} | "
                          f"Val Reward: {val_losses['reward']:.4f} | Val Done: {val_losses['done']:.4f}")
                    self.world_model.train()  # Set back to train mode after evaluation

                # Save model checkpoint
                if global_step > 0 and global_step % checkpoint_freq == 0:
                    model_state_to_save = self.world_model.module.state_dict() if isinstance(self.world_model,
                                                                                             nn.DataParallel) else self.world_model.state_dict()
                    filename = GRU_WM_CHECKPOINTS_DIR / f"world_model_step_{global_step}.pth"
                    torch.save(model_state_to_save, filename)
                    print(f"Saved model checkpoint at step {global_step}.")

                if profiler:
                    profiler.step()

        if profiler:
            profiler.stop()
            print("Profiling finished. You can view the trace with TensorBoard.")

        print("Training finished.")
        # Plot the final losses
        self.plot_losses()


if __name__ == "__main__":
    # Argument Parsing: Allow selecting config from command line
    parser = argparse.ArgumentParser(description="Train GRU World Model")
    parser.add_argument("--config", type=str, default="default",
                        help="Name of the configuration to use (e.g., 'default', 'test').")
    parser.add_argument("--save-data-to", type=str, default=None,
                        help="Path to save the collected data to. Data is not saved unless this is specified.")
    parser.add_argument("--load-data-from", type=str, default=None,
                        help="Path to load data from, skipping collection.")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to a model checkpoint to resume training.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")
    args = parser.parse_args()

    # Load the chosen configuration
    config = get_config(args.config)
    print(f"Loaded configuration: '{args.config}'")

    # Set multiprocessing start method - crucial for CUDA.
    try:
        mp.set_start_method('spawn')
    except RuntimeError as e:
        print(f"Could not set start method (possibly already set or not allowed): {e}")
        pass

    print(f"Starting GRU World Model training on device: {config['device']}")

    if "cuda" in str(config['device']):
        print(f"Using float32 matmul high precision for CUDA training.")
        torch.set_float32_matmul_precision('high')

    # checks for checkpoint files
    if not os.path.exists(VQ_VAE_CHECKPOINT_FILENAME):
        print(
            f"CRITICAL ERROR: VAE Checkpoint {VQ_VAE_CHECKPOINT_FILENAME} not found. Exiting before starting workers.")
        exit()

    sequence_data_buffer = None
    if args.load_data_from:
        if os.path.exists(args.load_data_from):
            print(f"Loading sequence data from {args.load_data_from}...")
            sequence_data_buffer = torch.load(args.load_data_from, map_location=config['device'])
            limit = config["num_steps"]
            if limit < len(sequence_data_buffer['actions']):
                # Limit the dataset to the first 'limit' sequences
                for key in sequence_data_buffer:
                    sequence_data_buffer[key] = sequence_data_buffer[key][:limit]
            print("Data loaded successfully.")
        else:
            print(f"ERROR: Data file not found at {args.load_data_from}. Exiting.")
            exit()
    else:
        # Collect Sequence Data (Parallelized)
        print(f"Number of collection workers configured: {config['num_collection_workers']}")
        start_collect_time = time.time()
        sequence_data_buffer = collect_sequences_for_gru(
            num_steps_total=config["num_steps"],
            device_str_main=config["device"],
            num_collection_workers_int=config["num_collection_workers"],
            env_name_str_for_worker=config["env_name"],
            max_episode_steps_collect_int=config["max_episode_steps_collect"]
        )
        print(f"Sequence data collection (parallel/serial) took {time.time() - start_collect_time:.2f} seconds.")

        if sequence_data_buffer and args.save_data_to:
            print(f"Saving collected data to {args.save_data_to}...")
            save_dir = os.path.dirname(args.save_data_to)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(sequence_data_buffer, args.save_data_to)
            print(f"Data saved to {args.save_data_to}.")

    if not sequence_data_buffer:
        print("ERROR: No sequence data collected. Exiting.")
        exit()

    # Prepare DataLoader with train/validation split
    print("Splitting data into training and validation sets...")
    full_dataset = SequenceDataset(sequence_data_buffer, config["sequence_length"])

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
        pin_memory=True if "cuda" in str(config['device']) else False  # Added pin_memory
    )
    val_dataloader = DataLoader(
        full_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=config["num_loader_workers"],
        pin_memory=True if "cuda" in str(config['device']) else False  # Added pin_memory
    )
    # --- End of new DataLoader code ---

    # Initialize GRU World Model
    world_model_gru = WorldModelGRU(
        latent_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
        dropout_rate=config['dropout_rate']  # Pass dropout_rate
    )
    world_model_gru.to(config['device'])

    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        print(f"Loading pre-trained model from {args.checkpoint_path}...")
        world_model_gru.load_state_dict(torch.load(args.checkpoint_path, map_location=config['device']))

    # Compile the GRU model
    world_model_gru = torch.compile(world_model_gru)

    # Initialize VQ-VAE Model
    vq_vae_model = VQVAE(embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS)
    vq_vae_model.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=config['device']))
    vq_vae_model.to(config['device'])
    vq_vae_model.eval()

    # Handle DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using nn.DataParallel for GRU model training across {torch.cuda.device_count()} GPUs.")
        world_model_gru = nn.DataParallel(world_model_gru)

    # Create the trainer instance
    # The trainer encapsulates the model, optimizer, and training logic.
    trainer = GruWorldModelTrainer(
        world_model_gru,
        vq_vae_model,
        config,
        train_dataloader,  # Pass train_dataloader
        val_dataloader  # Pass val_dataloader
    )

    # Run the training loop
    print("Starting GRU World Model training via WorldModelTrainer...")
    start_train_time = time.time()
    # trainer saves checkpoints automatically during training
    trainer.train(num_epochs=config["epochs"],
                  copy_vq_weights=args.checkpoint_path is None,
                  profile=args.profile)
    print(f"GRU World Model training took {time.time() - start_train_time:.2f} seconds.")

    # Save the final GRU World Model
    try:
        model_state_to_save = world_model_gru.module.state_dict() if isinstance(world_model_gru,
                                                                                nn.DataParallel) else world_model_gru.state_dict()
        torch.save(model_state_to_save, WM_CHECKPOINT_FILENAME_GRU)
        print(f"GRU World Model saved to {WM_CHECKPOINT_FILENAME_GRU}")
    except Exception as e:
        print(f"Error saving GRU World Model: {e}")
