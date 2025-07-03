# utils.py
import cv2
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import numpy as np

from vq_conv_vae import VQVAE, EMBEDDING_DIM, NUM_EMBEDDINGS

# --- Configuration Constants ---
ENV_NAME = "CarRacing-v3"
WM_HIDDEN_DIM = 256  # Hidden dimension for the World Model MLP
IMG_SIZE = 64  # Resize frames
# CHANNELS = 3  # RGB channels
CHANNELS = 1  # Grayscale channel
NUM_STACK = 4  # Number of latent vectors to stack
LATENT_DIM = 32  # Size of the latent space vector z
ACTION_DIM = 3  # CarRacing: Steering, Gas, Brake

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)  # Use GPU if available, else CPU

# --- File Paths ---
VAE_CHECKPOINT_FILENAME = f"checkpoints/{ENV_NAME}_cvae_ld{LATENT_DIM}_epoch10.pth"
VQ_VAE_CHECKPOINT_FILENAME = f"checkpoints/{ENV_NAME}_vqvae_ld{LATENT_DIM}.pth"
WM_MODEL_SUFFIX = f"ld{LATENT_DIM}_ac{ACTION_DIM}"
WM_CHECKPOINT_FILENAME = f"checkpoints/{ENV_NAME}_worldmodel_mlp_{WM_MODEL_SUFFIX}.pth"
WM_CHECKPOINT_FILENAME_GRU = f"checkpoints/{ENV_NAME}_worldmodel_gru_{WM_MODEL_SUFFIX}.pth"


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
            low=-np.inf,
            high=np.inf,
            shape=(num_stack, *original_shape),  # (num_stack, H, W, C)
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


class ActionClipWrapper(gym.ActionWrapper):
    """
    A wrapper that clips the actions passed to the environment to ensure they
    are within the valid action space bounds.

    This is particularly useful for continuous action spaces where the policy
    (e.g., a neural network) might output values slightly outside the
    defined [-1, 1] or other ranges.
    """

    def __init__(self, env: gym.Env):
        """
        Initializes the ActionClipWrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        # The action space itself is not changed by this wrapper.
        # We rely on self.action_space (which is self.env.action_space)
        # to provide the low and high bounds for clipping.
        # Ensure the environment has a Box action space for clipping to make sense.
        if not isinstance(self.env.action_space, spaces.Box):
            print(f"Warning: ActionClipWrapper is typically used with Box action spaces. "
                  f"The current action space is {self.env.action_space}.")

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Clips the given action to the bounds of the environment's action space.

        Args:
            action: The action to be clipped. Expected to be a NumPy array
                    compatible with the environment's action space.

        Returns:
            The clipped action as a NumPy array.
        """
        # self.action_space refers to the underlying environment's action space
        # as gym.ActionWrapper does not modify it by default.
        if isinstance(self.action_space, spaces.Box):
            # Ensure action is a numpy array if it's not already (e.g. if coming from PyTorch tensor)
            # However, typically the agent framework would convert to numpy before env.step()
            # For robustness, we can ensure it here if there's doubt.
            # if not isinstance(action, np.ndarray):
            #     action = np.array(action) # This line can be risky if types are unexpected

            clipped_action = np.clip(
                action,
                self.action_space.low,
                self.action_space.high
            )
            return clipped_action
        else:
            # This case should ideally not be reached if wrappers are correctly stacked.
            # If action_space is not Box, clipping is ill-defined.
            print(
                f"Warning: Action space for ActionClipWrapper is not Box ({self.action_space}), returning action unclipped.")
            return action


class VaeEncodeWrapper(gym.ObservationWrapper):
    def __init__(self, env, vq_vae_model: torch.nn.Module, device: torch.device, save_latent_for_render=False):
        """
        A wrapper that encodes the raw observation from the environment into a quantized latent vector
        using a VQ-VAE model. The observation space is transformed to a single flattened latent vector.
        """
        super().__init__(env)
        self.vq_vae_model = vq_vae_model.to(device).eval()
        self.device = device
        self.save_latent_for_render = save_latent_for_render

        # Determine the shape of the flattened quantized output for a single frame
        dummy_input = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE).to(self.device)
        with torch.no_grad():
            z_continuous = self.vq_vae_model.encoder(dummy_input)
            _, quantized_sample, _ = self.vq_vae_model.vq_layer(z_continuous)

        # The new observation space is a single flattened latent vector
        flat_latent_shape = (np.prod(quantized_sample.shape[1:]),)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=flat_latent_shape,
            dtype=np.float32
        )

        self.last_quantized_latent_for_render = None  # For rendering purposes

    def observation(self, obs_raw_numpy):
        # Preprocess the raw frame
        processed_frame = preprocess_observation(obs_raw_numpy)
        processed_tensor = torch.tensor(processed_frame, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
        processed_tensor = processed_tensor.unsqueeze(0).to(self.device)  # Add batch dim

        # Encode to get the quantized latent vector
        with torch.no_grad():
            z_continuous = self.vq_vae_model.encoder(processed_tensor)
            _, quantized, _ = self.vq_vae_model.vq_layer(z_continuous)

        if self.save_latent_for_render:
            self.last_quantized_latent_for_render = quantized.clone()  # Store for rendering

        # Flatten and return as a numpy array
        return quantized.reshape(-1).cpu().numpy()


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


def make_env_sb3(
        env_id: str,
        vq_vae_model_instance: torch.nn.Module,  # Pass the loaded VAE model
        frame_stack_num: int,
        device_for_vae: torch.device,
        gamma: float,
        render_mode: str = None,
        max_episode_steps: int = None,
        save_latent_for_render: bool = False,
        seed: int = 0  # Seed for reproducibility
):
    """
    Creates and wraps the environment for use with Stable Baselines3.
    The VAE model instance must be passed.
    """
    # Create the base environment
    # For Gymnasium 0.26+, max_episode_steps is part of gym.make()
    # For older versions, it might be applied via a TimeLimit wrapper later if not None.
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

    # ActionClipWrapper:
    #    - Clips actions received from the agent (which are in [-1, 1]^3 as per ActionTransformWrapper's space)
    #    - This ensures actions are strictly within the [-1, 1]^3 bounds before transformation by ActionTransformWrapper.
    #    - The ActionTransformWrapper then does its own clipping to the *underlying* environment's true bounds.
    env = ActionClipWrapper(env)  # This will clip to the action_space defined by ActionTransformWrapper

    # --- Observation Wrappers ---
    # FrameSkip: Skips frames to reduce data size and speed up training and inference.
    env = FrameSkip(env, skip=4)  # Skip every 4th frame

    # VaeEncodeWrapper: Preprocess and encode raw frame into quantized latent vectors using VQ-VAE and returns a single flattened latent vector
    env = VaeEncodeWrapper(env, vq_vae_model_instance, device_for_vae, save_latent_for_render)

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


print(f"Utils loaded. Using device: {DEVICE}")
print(f"VQ-VAE Path: {VQ_VAE_CHECKPOINT_FILENAME}")
print(f"WM Path: {WM_CHECKPOINT_FILENAME}")


def _init_env_fn_sb3(rank: int, seed: int = 0, config_env_params: dict = None):
    """
    Creates an environment instance for SubprocVecEnv or DummyVecEnv.
    Each process/environment will call this function.
    """
    if config_env_params is None:
        config_env_params = {}

    set_random_seed(seed + rank)  # Ensure each environment has a different seed

    vae_device_for_subprocess = torch.device(DEVICE_STR)

    # print(f"Rank {rank}: Attempting to load VAE on device: {vae_device_for_subprocess}")

    vq_vae_model = VQVAE(in_channels=CHANNELS, embedding_dim=EMBEDDING_DIM, num_embeddings=NUM_EMBEDDINGS).to(
        vae_device_for_subprocess)
    vq_vae_checkpoint_path = VQ_VAE_CHECKPOINT_FILENAME

    try:
        vq_vae_model.load_state_dict(torch.load(vq_vae_checkpoint_path, map_location=vae_device_for_subprocess))
        vq_vae_model.eval()
        # print(f"Rank {rank}: Successfully loaded VAE from {vae_checkpoint_path} to {vae_device_for_subprocess}")
    except FileNotFoundError:
        print(f"Rank {rank}: ERROR: VAE checkpoint '{vq_vae_checkpoint_path}' not found. Train VAE first.")
        raise
    except Exception as e:
        print(f"Rank {rank}: ERROR loading VAE: {e}")
        raise

    env = make_env_sb3(
        env_id=config_env_params.get("env_name_config", ENV_NAME),
        vq_vae_model_instance=vq_vae_model,
        frame_stack_num=config_env_params.get("num_stack_config", NUM_STACK),
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
