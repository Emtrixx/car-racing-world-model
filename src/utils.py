# utils.py
from collections import deque
from typing import Optional, Any
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.dream_env import GruDreamEnv
from src.dream_env_transformer import DreamEnvTransformer
from src.vq_conv_vae import VQVAE_EMBEDDING_DIM, VQVAE_NUM_EMBEDDINGS, VQVAE
from src.world_model import D_MODEL, GRU_NUM_LAYERS, WorldModelGRU
from src.transformer_world_model import WorldModelTransformer

# --- Configuration Constants ---
ENV_NAME = "CarRacing-v3"
NUM_STACK = 4  # Number of latent vectors to stack
ACTION_DIM = 3  # CarRacing: Steering, Gas, Brake

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)  # Use GPU if available, else CPU

# --- Directory Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SB3_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "sb3_checkpoints"
VQVAE_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "vqvae_checkpoints"
GRU_WM_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "gru_wm_checkpoints"
TRANSFORMER_WM_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "transformer_wm_checkpoints"
VIDEO_DIR = PROJECT_ROOT / "videos"
IMAGES_DIR = PROJECT_ROOT / "images"
ASSETS_DIR = PROJECT_ROOT / "assets"  # used in webapp for visualization
DATA_DIR = PROJECT_ROOT / "data"  # For storing datasets, e.g., frames for VAE training
LOGS_DIR = PROJECT_ROOT / "logs"
SB3_LOG_DIR = LOGS_DIR / "sb3_logs"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
SB3_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
VQVAE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
GRU_WM_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORMER_WM_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- File Paths ---
# VQ_VAE_CHECKPOINT_FILENAME = VQVAE_CHECKPOINTS_DIR / f"{ENV_NAME}_vqvae_ld{256}_cb{512}.pth"  # latent_dim=256, codebook_size=512
VQ_VAE_CHECKPOINT_FILENAME = VQVAE_CHECKPOINTS_DIR / f"{ENV_NAME}_vqvae_ld{VQVAE_EMBEDDING_DIM}_cb{VQVAE_NUM_EMBEDDINGS}.pth"
VQ_VAE_CHECKPOINT_BEST_FILENAME = VQVAE_CHECKPOINTS_DIR / f"{ENV_NAME}_vqvae_ld{VQVAE_EMBEDDING_DIM}_cb{VQVAE_NUM_EMBEDDINGS}_best.pth"
WM_MODEL_SUFFIX_GRU = f"ld{VQVAE_EMBEDDING_DIM}_ac{ACTION_DIM}_dm{D_MODEL}_ly{GRU_NUM_LAYERS}"
WM_CHECKPOINT_FILENAME_GRU = GRU_WM_CHECKPOINTS_DIR / f"{ENV_NAME}_worldmodel_gru_{WM_MODEL_SUFFIX_GRU}.pth"
# WM_CHECKPOINT_FILENAME_GRU = GRU_WM_CHECKPOINTS_DIR / "world_model_step_45000.pth"
# WM_CHECKPOINT_FILENAME_GRU = GRU_WM_CHECKPOINTS_DIR / "dyn_default_wm_step_1835008.pth"
WM_CHECKPOINT_FILENAME_GRU = GRU_WM_CHECKPOINTS_DIR / "CarRacing-v3_worldmodel_gru_ld256_ac3_dm1024_ly3.pth"
# WM_CHECKPOINT_FILENAME_TRANSFORMER = TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_world_model_default.pth"
# WM_CHECKPOINT_FILENAME_TRANSFORMER = TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_world_model_t_btf_default.pth"


# WM_CHECKPOINT_FILENAME_TRANSFORMER = TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_world_model_test.pth"
# WM_CHECKPOINT_FILENAME_TRANSFORMER = TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_wm_test.pth"
WM_CHECKPOINT_FILENAME_TRANSFORMER = TRANSFORMER_WM_CHECKPOINTS_DIR / f"transformer_wm_step_40000.pth"


# WM_CHECKPOINT_FILENAME_TRANSFORMER = TRANSFORMER_WM_CHECKPOINTS_DIR / f"dyn_default_wm_step_958464.pth"
# WM_CHECKPOINT_FILENAME_TRANSFORMER = TRANSFORMER_WM_CHECKPOINTS_DIR / f"dyn_default_carracing-v3_wm_final.pth"


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
    # gray_obs = cv2.cvtColor(cropped_obs, cv2.COLOR_RGB2GRAY)

    # 3. Resizing
    resized_obs = cv2.resize(cropped_obs, resize_dim, interpolation=cv2.INTER_AREA)

    # 4. Normalization
    normalized_obs = resized_obs / 255.0

    # Add channel dimension for consistency (e.g., for TensorFlow or PyTorch)
    # return normalized_obs.reshape(resize_dim[0], resize_dim[1], 1)
    return normalized_obs.astype(np.float32)  # Ensure float32 dtype


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


class OffTrackRewardShaper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Heavily penalize going off-track
        if np.mean(obs[:, :, 1]) > 180.0:  # A simple check for green grass
            reward -= 8.0  # Increase the off-track penalty significantly

        return obs, reward, terminated, truncated, info


def make_env_sb3(
        env_id: str,
        frame_stack_num: int,
        gamma: float = 0.99,  # Discount factor for rewards
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

    # --- SkipStartFramesWrapper: Skips the first N frames at the start of each episode. @CarRacing-v3
    env = SkipStartFramesWrapper(env, skip=50)  # Skip the first 50 frames

    # --- Observation Wrappers ---
    # FrameSkip: Skips frames to reduce data size and speed up training and inference.
    env = FrameSkip(env, skip=4)  # Skip every 4th frame

    # OffTrackRewardShaper: Heavily penalizes going off-track. @CarRacing-v3
    # env = OffTrackRewardShaper(env)

    # VaeEncodeWrapper: Preprocess raw frame (crop, resize)
    env = PreprocessWrapper(env)  # @CarRacing-v3

    # FrameStackWrapper: Stacks the last N observations (quantized latent vectors) to create a temporal context.
    env = FrameStackWrapper(env, frame_stack_num)

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


def _create_dream_env(config, wm_state_dict, vq_vae_state_dict, start_state_pool, seed, world_model_type):
    """
    A pickleable function to create the dream environment in a subprocess.
    It reconstructs the models from their state_dicts and uses a pre-computed pool of start states.
    """
    device = torch.device(config['device'])

    # Reconstruct VQ-VAE
    vq_vae = VQVAE()
    vq_vae.load_state_dict(vq_vae_state_dict)
    vq_vae = vq_vae.to(device)
    vq_vae.eval()
    vq_vae = torch.compile(vq_vae)

    if "cuda" in str(config['device']):
        print(f"Using float32 matmul high precision for CUDA training.")
        torch.set_float32_matmul_precision('high')

    if world_model_type == 'transformer':
        world_model = WorldModelTransformer(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            ff_dim=config['ff_dim'],
            grid_size=config['grid_size'],
            dropout_rate=config['dropout_rate'],
            max_seq_len=config['max_seq_len'],
            action_dim=config['action_dim'],
            codebook_size=config['codebook_size'],
            vqvae_embed_dim=config['vqvae_embed_dim']
        ).to(device)
        env_class = DreamEnvTransformer
        env_kwargs = {'history_length': config['history_length']}
    elif world_model_type == 'gru':
        world_model = WorldModelGRU(
            latent_dim=config['vqvae_embed_dim'],
            action_dim=config['action_dim'],
            d_model=config['gru_d_model'],
            gru_num_layers=config['gru_num_layers'],
            codebook_size=config['codebook_size'],
            grid_size=config['grid_size']
        ).to(device)
        env_class = GruDreamEnv
        env_kwargs = {}
    else:
        raise ValueError(f"Unknown world model type: {world_model_type}")

    world_model = torch.compile(world_model)  # Compile for performance
    world_model.load_state_dict(wm_state_dict)
    world_model.eval()

    dream_env = env_class(
        world_model=world_model,
        vq_vae=vq_vae,
        device=device,
        seed=seed,
        horizon=config['dream_horizon'],
        start_state_pool=start_state_pool,
        **env_kwargs
    )
    return FrameStackWrapper(dream_env, num_stack=NUM_STACK)


def fmt_int(n: int) -> str:
    # 1_000_000 style grouping
    return f"{int(n):_}"
