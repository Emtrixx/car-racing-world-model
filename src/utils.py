# utils.py
import os
from collections import deque
from typing import Optional, Any
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch import nn
from torch.optim import Adam

# --- Configuration Constants ---
ENV_NAME = "CarRacing-v3"
WM_HIDDEN_DIM = 256  # Hidden dimension for the World Model MLP
NUM_STACK = 4  # Number of latent vectors to stack
LATENT_DIM = 32  # Size of the latent space vector z
ACTION_DIM = 3  # CarRacing: Steering, Gas, Brake

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)  # Use GPU if available, else CPU

# --- Directory Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SB3_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "sb3_checkpoints"
VQVAE_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "vqvae_checkpoints"
GRU_WM_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "gru_wm_checkpoints"
VIDEO_DIR = PROJECT_ROOT / "videos"
IMAGES_DIR = PROJECT_ROOT / "images"
ASSETS_DIR = PROJECT_ROOT / "assets"  # used in webapp for visualization
DATA_DIR = PROJECT_ROOT / "data"  # For storing datasets, e.g., frames for VAE training
LOGS_DIR = PROJECT_ROOT / "logs"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
SB3_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
VQVAE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
GRU_WM_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- File Paths ---
VQ_VAE_CHECKPOINT_FILENAME = VQVAE_CHECKPOINTS_DIR / f"{ENV_NAME}_vqvae_ld{64}.pth"
WM_MODEL_SUFFIX = f"ld{LATENT_DIM}_ac{ACTION_DIM}"
WM_CHECKPOINT_FILENAME_GRU = GRU_WM_CHECKPOINTS_DIR / f"{ENV_NAME}_worldmodel_gru_{WM_MODEL_SUFFIX}.pth"


# --- Preprocessing Function ---
def preprocess_observation(obs, resize_dim=(64, 64)):
    """
    Applies preprocessing steps to a raw observation from the CarRacing-v3 environment.

    Args:
        obs (np.ndarray): A raw observation from the environment (96x96x3 RGB).
        resize_dim (tuple): The target dimensions (height, width) for resizing.

    Returns:
        np.ndarray: The preprocessed observation (height, width, 1) with values in [0, 1].
    """
    # 1. Crop the bottom HUD
    # The HUD is in the bottom 12 pixels of the 96x96 image.
    cropped_obs = obs[:-12, :, :]

    # 2. Grayscaling
    gray_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2GRAY)

    # 3. Resizing
    resized_obs = cv2.resize(gray_obs, resize_dim, interpolation=cv2.INTER_AREA)

    # 4. Normalization
    normalized_obs = resized_obs / 255.0

    # Add channel dimension for consistency (e.g., for TensorFlow or PyTorch)
    return normalized_obs.reshape(resize_dim[0], resize_dim[1], 1)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=NUM_STACK):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        original_shape = self.env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_stack, *original_shape),  # (num_stack, height, width, channels)
            dtype=self.env.observation_space.dtype
        )

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, "Not enough frames in buffer"
        return np.array(list(self.frames), dtype=self.env.observation_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)  # Repeat first frame N times
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info


class PreprocessWrapper(gym.ObservationWrapper):
    """
    A wrapper that preprocesses the raw observation from the CarRacing-v3 environment
    """

    def __init__(self, env):
        super().__init__(env)
        dummy_input = np.zeros((93, 93, 3), dtype=np.uint8)  # Dummy input for preprocessing
        processed_frame = preprocess_observation(dummy_input)
        shape = processed_frame.shape

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=shape,  # (height, width, channels)
            dtype=np.float32
        )

    def observation(self, obs_raw_numpy):
        # Preprocess the raw frame
        processed_frame = preprocess_observation(obs_raw_numpy)
        return processed_frame


# For SB3
class ActionTransformWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # The action space of THIS wrapper (what the SB3 agent sees)
        # It's Box(-1, 1) for all 3 components (steering, gas control, brake control)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        # Store the original action space of the wrapped environment for clipping reference
        self._underlying_action_space = env.action_space  # This is CarRacing's original action space

    def action(self, action_from_agent: np.ndarray) -> np.ndarray:
        # action_from_agent comes from SB3 agent, in range [-1, 1] for all components. Shape: (ACTION_DIM,)
        # For CarRacing-v3, ACTION_DIM is 3: [steer_control, gas_control, brake_control]

        steer_control = action_from_agent[0]
        gas_control = action_from_agent[1]  # Represents gas intensity, agent outputs in [-1, 1]
        brake_control = action_from_agent[2]  # Represents brake intensity, agent outputs in [-1, 1]

        # Transform to CarRacing's native action ranges:
        # Steering: agent's output is already in [-1, 1]
        actual_steering = steer_control
        # Gas: agent's output in [-1, 1] needs to be mapped to [0, 1]
        actual_gas = (gas_control + 1.0) / 2.0
        # Brake: agent's output in [-1, 1] needs to be mapped to [0, 1]
        actual_brake = (brake_control + 1.0) / 2.0

        transformed_action = np.array([actual_steering, actual_gas, actual_brake], dtype=np.float32)

        # Clip to the *underlying* environment's actual valid range.
        # This ensures that even if the transformation logic is slightly off or due to float precision,
        # the action sent to the base CarRacing environment is valid.
        clipped_transformed_action = np.clip(transformed_action,
                                             self._underlying_action_space.low,
                                             self._underlying_action_space.high)
        return clipped_transformed_action


class FrameSkip(gym.Wrapper):
    """
    Return only every `skip`-th frame and repeat the same action `skip` times.
    The reward is the sum of rewards over the skipped frames.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        if skip <= 0:
            raise ValueError(f"Frame skip must be a positive integer, got {skip}")
        self._skip = skip

    def step(self, action):
        """
        Repeat action, sum reward, and return the last observation.
        """
        total_reward = 0.0

        for _ in range(self._skip):
            # The core of the frame skip
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # If the episode ends, stop skipping and return the last state
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # The reset is not affected by frame skipping
        return self.env.reset(**kwargs)


class SkipStartFramesWrapper(gym.Wrapper):
    """
    A wrapper that automatically skips a specified number of frames
    at the beginning of each episode by performing a no-op action.
    """

    def __init__(self, env: gym.Env, skip: int = 50):
        """
        Initializes the wrapper.

        Args:
            env: The environment to wrap.
            skip (int): The number of frames to skip on reset.
        """
        super().__init__(env)
        if skip < 1:
            raise ValueError("The number of frames to skip must be at least 1.")
        self.skip = skip
        # Define a no-op action
        self.noop_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[
        Any, dict[str, Any]]:
        """
        Resets the environment and then skips the specified number of frames.
        """
        # First, reset the underlying environment as usual
        obs, info = self.env.reset(seed=seed, options=options)

        # Now, loop for the specified number of steps, performing a no-op action
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)

            # Important edge case: If the episode ends during the skip
            # (e.g., in very short/fast environments), we must reset again
            # to ensure the agent starts from a valid, non-terminal state.
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)

        # Return the observation and info from the final skipped frame
        return obs, info


def make_env_sb3(
        env_id: str,
        frame_stack_num: int,
        gamma: float,
        render_mode: str = None,
        max_episode_steps: int = None,
        seed: int = 0  # Seed for reproducibility
):
    """
    Creates and wraps the environment for use with Stable Baselines3.
    The VAE model instance must be passed.
    """
    # Create the base environment
    env_kwargs = {}
    if render_mode:
        env_kwargs['render_mode'] = render_mode
    if max_episode_steps:  # Gymnasium 0.26+
        env_kwargs['max_episode_steps'] = max_episode_steps

    env = gym.make(env_id, **env_kwargs)

    # Apply seed. Important for reproducibility, especially with VecEnv.
    # Note: env.reset(seed=seed) is preferred in Gymnasium 0.26+ for initial seeding.
    # For continuous seeding of action_space/observation_space, it's more complex with wrappers.
    # SubprocVecEnv handles seeding of each env instance using the seed passed to its env_fn.
    # So, the primary seeding point will be in the env_fn for SubprocVecEnv.
    # However, calling reset here with a seed is good practice for the initial state.
    # obs, info = env.reset(seed=seed) # Initial reset with seed

    # --- Action Wrappers ---
    # ActionTransformWrapper:
    #    - Agent outputs actions in [-1, 1]^3.
    #    - This wrapper transforms them to CarRacing's native ranges
    #    - It defines self.action_space = Box([-1, 1]^3, ...) which SB3 will see.
    env = ActionTransformWrapper(env)

    # --- SkipStartFramesWrapper: Skips the first N frames at the start of each episode.
    env = SkipStartFramesWrapper(env, skip=50)  # Skip the first 50 frames

    # --- Observation Wrappers ---
    # FrameSkip: Skips frames to reduce data size and speed up training and inference.
    env = FrameSkip(env, skip=4)  # Skip every 4th frame

    # VaeEncodeWrapper: Preprocess raw frame (crop, grayscale, resize)
    env = PreprocessWrapper(env)

    # FrameStackWrapper: Stacks the last N observations (quantized latent vectors) to create a temporal context.
    env = FrameStackWrapper(env, frame_stack_num)

    # --- Reward Wrapper ---
    # NormalizeReward: Normalizes rewards.
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)

    # For SB3, RecordEpisodeStatistics is often useful when using VecEnvs for logging.
    # It should be one of the outermost wrappers if used.
    # env = gym.wrappers.RecordEpisodeStatistics(env) # Add this if you want SB3 to log ep_len_mean, ep_rew_mean

    # print(f"Seed {seed}: Final wrapped environment observation space: {env.observation_space}")
    # print(f"Seed {seed}: Sample observation shape: {env.observation_space.sample().shape}")
    # print(f"Seed {seed}: Final wrapped environment action space: {env.action_space}")
    # print(f"Seed {seed}: Sample action: {env.action_space.sample()}")
    return env


def _init_env_fn_sb3(rank: int, seed: int = 0, config_env_params: dict = None):
    """
    Creates an environment instance for SubprocVecEnv or DummyVecEnv.
    Each process/environment will call this function.
    """
    if config_env_params is None:
        config_env_params = {}

    set_random_seed(seed + rank)  # Ensure each environment has a different seed

    env = make_env_sb3(
        env_id=config_env_params.get("env_name_config", ENV_NAME),
        frame_stack_num=config_env_params.get("num_stack_config", NUM_STACK),
        gamma=config_env_params.get("gamma_config", 0.99),
        render_mode=config_env_params.get("render_mode", None),
        max_episode_steps=config_env_params.get("max_episode_steps_config", 1000),
        seed=seed + rank  # Pass seed to make_env_sb3 for its own seeding logic if any
    )
    # Monitor wrapper is important for SB3 to log episode rewards and lengths,
    # especially when using DummyVecEnv or if RecordEpisodeStatistics is not used inside make_env_sb3.
    env = Monitor(env)
    return env


class WorldModelDataCollector:
    """
    Collects training data for the World Model by running a pretrained policy.
    """

    def __init__(self, env, ppo_agent, vq_vae_model, device):
        self.env = env
        self.ppo_agent = ppo_agent
        self.vq_vae_model = vq_vae_model
        self.device = device
        # Use a simple deque as a replay buffer
        self.replay_buffer = deque(maxlen=250_000)

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
            _, _, indices = self.vq_vae_model.vq_layer(z_continuous)

        return indices.view(1, -1)  # Flatten to [1, 16]

    def collect_steps(self, num_steps: int):
        """
        Runs the PPO agent in the environment for a given number of steps.

        Args:
            num_steps (int): The total number of steps to collect.
        """
        print(f"Collecting {num_steps} steps of experience...")
        obs, _ = self.env.reset()

        for step in range(num_steps):
            # Get action from the pretrained PPO agent
            action, _ = self.ppo_agent.predict(obs, deterministic=False)

            # Step the environment with the action
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Get the ground truth token indices for the next observation
            next_state_tokens = self.get_vq_indices(next_obs[-1])  # access last frame in the stack

            # Store the relevant data tuple for world model training
            # We store the action, reward, done flag, and the tokenized *next* state.
            self.replay_buffer.append({
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.float32),
                "next_tokens": next_state_tokens.squeeze(0).to(torch.int64)  # Store as [16]
            })

            # Update observation and reset if the episode is over
            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()
                print(f"Episode finished. Buffer size: {len(self.replay_buffer)}")

        print(f"Collection complete. Final buffer size: {len(self.replay_buffer)}")


class WorldModelTrainer:
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
                if isinstance(self.world_model, nn.DataParallel):
                    hidden_state = self.world_model.module.get_initial_hidden_state(batch_size, self.device)
                else:
                    hidden_state = self.world_model.get_initial_hidden_state(batch_size, self.device)

                seq_token_loss, seq_reward_loss, seq_done_loss = 0, 0, 0
                sequence_length = batch['actions'].size(1)

                for t in range(sequence_length):
                    action_t = batch['actions'][:, t]
                    ground_truth_tokens_t = batch['next_tokens'][:, t]
                    ground_truth_reward_t = batch['rewards'][:, t]
                    ground_truth_done_t = batch['dones'][:, t]

                    pred_logits, pred_reward, pred_done_logits, next_hidden_state = self.world_model(
                        action_t, hidden_state, ground_truth_tokens=ground_truth_tokens_t
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
                    hidden_state = next_hidden_state

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

    def train(self, num_epochs):  # Removed dataloader argument
        """Main training loop that iterates over a DataLoader."""
        print("Starting world model training...")
        if isinstance(self.world_model, nn.DataParallel):
            self.world_model.module.token_embedding.weight.data.copy_(
                self.vq_vae_model.vq_layer.embedding.weight.data
            )
        else:
            self.world_model.token_embedding.weight.data.copy_(
                self.vq_vae_model.vq_layer.embedding.weight.data
            )
        print("Copied VQ-VAE weights to world model token embedding.")

        self.world_model.train()

        # Initialize step counters
        global_step = 0
        total_train_steps = len(self.train_dataloader) * num_epochs  # Use self.train_dataloader
        log_freq = self.config.get('log_freq', 100)
        val_freq = self.config.get('val_freq', log_freq * 5)  # val_freq from config
        checkpoint_freq = self.config.get('checkpoint_freq', 1000)

        for epoch in range(1, num_epochs + 1):
            for batch_idx, batch in enumerate(self.train_dataloader):  # Use self.train_dataloader
                global_step += 1

                for key in batch:
                    batch[key] = batch[key].to(self.device)

                # Initialize hidden state for the start of the sequences
                batch_size = batch['actions'].size(0)
                hidden_state = self.world_model.get_initial_hidden_state(batch_size, self.device) \
                    if not isinstance(self.world_model, nn.DataParallel) \
                    else self.world_model.module.get_initial_hidden_state(batch_size, self.device)

                total_token_loss, total_reward_loss, total_done_loss = 0, 0, 0
                sequence_length = batch['actions'].size(1)

                for t in range(sequence_length):
                    action_t = batch['actions'][:, t]
                    ground_truth_tokens_t = batch['next_tokens'][:, t]
                    ground_truth_reward_t = batch['rewards'][:, t]
                    ground_truth_done_t = batch['dones'][:, t]

                    pred_logits, pred_reward, pred_done_logits, next_hidden_state = self.world_model(
                        action_t, hidden_state, ground_truth_tokens=ground_truth_tokens_t
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
                    hidden_state = next_hidden_state

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
                    torch.save(model_state_to_save, f"world_model_step_{global_step}.pth")
                    print(f"Saved model checkpoint at step {global_step}.")

        print("Training finished.")
        # Plot the final losses
        self.plot_losses()
