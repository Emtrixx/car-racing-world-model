# utils.py
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torchvision import transforms
import gymnasium as gym
from gymnasium import spaces, Wrapper
from collections import deque
import numpy as np

from vq_conv_vae import VQVAE, EMBEDDING_DIM, NUM_EMBEDDINGS

# --- Configuration Constants ---
ENV_NAME = "CarRacing-v3"
WM_HIDDEN_DIM = 256  # Hidden dimension for the World Model MLP
IMG_SIZE = 64  # Resize frames
CHANNELS = 3  # RGB channels
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

# --- Data Preprocessing Transform (Identical for all parts) ---
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # Convert PIL Image to tensor (C, H, W) and scales to [0, 1]
])


# --- Helper Function to Preprocess and Encode Observation ---
def preprocess_and_encode(obs, transform_fn, vae_model, device):
    """
    Applies transform and encodes observation using the VAE encoder's mean.

    Args:
        obs (np.array): Raw observation from environment.
        transform_fn (callable): The preprocessing transform.
        vae_model (nn.Module): The loaded VAE model (in eval mode).
        device (torch.device): The target device.

    Returns:
        torch.Tensor: Latent state vector z (mean) on the specified device.
                      Shape: (LATENT_DIM)
    """
    processed_obs = transform_fn(obs).unsqueeze(0).to(device)  # Add batch dim and move to device
    with torch.no_grad():  # We don't need gradients for VAE encoding
        mu, logvar = vae_model.encode(processed_obs)
        # Using the mean (mu) is common for downstream tasks
        z = mu  # Shape: (1, LATENT_DIM)
    return z.squeeze(0)  # Remove batch dim -> Shape: (LATENT_DIM)


# --- Helper Function to Preprocess and Encode Observation Stack ---
def preprocess_and_encode_stack(
        raw_frame_stack,  # NumPy array from FrameStackWrapper: (num_stack, H, W, C)
        transform_fn,  # Your existing torchvision transform
        vae_model,  # Your loaded VAE model (in eval mode)
        device
):
    latent_vectors = []
    for i in range(raw_frame_stack.shape[0]):
        # same logic as for single frame
        raw_frame = raw_frame_stack[i]
        processed_frame = transform_fn(raw_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = vae_model.encode(processed_frame)
            latent_vectors.append(mu.squeeze(0))
    concatenated_latents = torch.cat(latent_vectors, dim=0)
    return concatenated_latents


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_stack=NUM_STACK):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        original_shape = self.env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
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


class LatentStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, vae_model, transform_fn, latent_dim, num_stack, device):
        super().__init__(env)
        self.vae_model = vae_model.to(device).eval()
        self.transform_fn = transform_fn
        self.latent_dim = latent_dim
        self.num_stack = num_stack
        self.device = device

        # New observation space is the concatenated latent vectors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_stack * self.latent_dim,),
            dtype=np.float32
        )

    def observation(self, obs_stack_raw_numpy):
        # obs_stack_raw_numpy comes from FrameStackWrapper: (num_stack, H, W, C) dtype=uint8
        latent_vectors_list = []
        for i in range(obs_stack_raw_numpy.shape[0]):  # Iterate through N frames in the stack
            raw_frame = obs_stack_raw_numpy[i]  # Single raw frame (H, W, C)

            # Apply torchvision transform (expects PIL or HWC NumPy, outputs Tensor C,H,W)
            processed_frame_tensor = self.transform_fn(raw_frame)  # Should be (C, H, W)

            # Add batch dim, move to device for VAE
            processed_frame_tensor = processed_frame_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                mu, _ = self.vae_model.encode(processed_frame_tensor)  # mu shape (1, LATENT_DIM)
                latent_vectors_list.append(mu.squeeze(0))  # Squeeze to (LATENT_DIM), keep on device

        # Concatenate the N latent vectors (still on device)
        concatenated_latents_tensor = torch.cat(latent_vectors_list, dim=0)  # Shape: (num_stack * LATENT_DIM,)

        # Environment observations should be NumPy arrays on CPU
        return concatenated_latents_tensor.cpu().numpy()


class LatentStateWrapperVQ(gym.ObservationWrapper):
    def __init__(self, env, vq_vae_model: torch.nn.Module, transform_fn, num_stack: int, device: torch.device):
        super().__init__(env)
        self.vq_vae_model = vq_vae_model.to(device).eval()
        self.transform_fn = transform_fn
        self.num_stack = num_stack
        self.device = device

        # Determine the shape of the flattened quantized output for a single frame
        # by doing a dummy forward pass.
        # Ensure IMG_CHANNELS, IMG_SIZE are correctly defined where this wrapper is used.
        # These should match the input expected by your VQVAE and transform_fn.
        dummy_input = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE).to(self.device)
        with torch.no_grad():
            z_continuous = self.vq_vae_model.encoder(dummy_input)
            # _, quantized_sample, _, encoding_indices_sample = self.vq_vae_model.vq_layer(z_continuous)
            # The vq_layer.forward returns: vq_loss, quantized_latents, encoding_indices
            _, quantized_sample, _ = self.vq_vae_model.vq_layer(z_continuous)

        # quantized_sample has shape (1, embedding_dim, H_feat, W_feat)
        # We will flatten the (embedding_dim, H_feat, W_feat) part
        self.flat_quantized_dim_per_frame = np.prod(quantized_sample.shape[1:])

        # New observation space is the concatenated flattened quantized vectors
        new_obs_shape = (self.num_stack * self.flat_quantized_dim_per_frame,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=new_obs_shape,
            dtype=np.float32  # Quantized vectors are floats
        )
        print(f"LatentStateWrapperVQ: Initialized with observation space shape: {new_obs_shape}")
        self.last_quantized_latent_for_render = None

    def observation(self, obs_stack_raw_numpy):
        # obs_stack_raw_numpy comes from FrameStackWrapper: (num_stack, H, W, C) dtype=uint8
        processed_quantized_vectors_list = []
        for i in range(obs_stack_raw_numpy.shape[0]):  # Iterate through N frames in the stack
            raw_frame = obs_stack_raw_numpy[i]  # Single raw frame (H, W, C)

            # Apply torchvision transform
            processed_frame_tensor = self.transform_fn(raw_frame)

            # Add batch dim, move to device for VQ-VAE
            processed_frame_tensor = processed_frame_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                # 1. Encode the frame
                z_continuous = self.vq_vae_model.encoder(processed_frame_tensor)
                # 2. Get the quantized representation from the VQ layer
                # vq_loss, quantized_single_frame, encoding_indices = self.vq_vae_model.vq_layer(z_continuous)
                # We only need the quantized_single_frame for the observation
                _, quantized_single_frame, _ = self.vq_vae_model.vq_layer(z_continuous)

                # quantized_single_frame has shape (1, embedding_dim, H_feat, W_feat)
                # Store a clone for rendering before flattening
                self.last_quantized_latent_for_render = quantized_single_frame.clone()

                # Flatten it to (embedding_dim * H_feat * W_feat)
                flat_quantized_vector = quantized_single_frame.reshape(-1)
                processed_quantized_vectors_list.append(flat_quantized_vector)

        # Concatenate the N flattened quantized latent vectors (still on device)
        # Each element in the list is a tensor of shape (flat_quantized_dim_per_frame,)
        concatenated_quantized_tensor = torch.cat(processed_quantized_vectors_list, dim=0)

        # Environment observations should be NumPy arrays on CPU
        return concatenated_quantized_tensor.cpu().numpy()


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


def make_env(vae_model_instance,  # The loaded and initialized VAE model
             transform_function,  # torchvision.transforms.Compose object
             env_id=ENV_NAME,
             frame_stack_num=NUM_STACK,
             render_mode=None,
             gamma=0.99,
             single_latent_dim=LATENT_DIM,
             device_for_vae=DEVICE,
             max_episode_steps=None,
             ):
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = ActionClipWrapper(env)  # Clip actions from agent before they hit the core env logic
    env = FrameStackWrapper(env, frame_stack_num)  # add frame stacking to observation
    env = LatentStateWrapper(env, vae_model_instance, transform_function,
                             single_latent_dim, frame_stack_num, device_for_vae)
    # env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -1.0, 1.0)) # clip reward
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)  # normalize reward

    print(f"Final wrapped environment observation space: {env.observation_space}")
    print(f"Sample observation shape from final env: {env.observation_space.sample().shape}")

    return env


def make_env_sb3(
        env_id: str,
        vq_vae_model_instance: torch.nn.Module,  # Pass the loaded VAE model
        transform_function,
        frame_stack_num: int,
        device_for_vae: torch.device,
        gamma: float,
        render_mode: str = None,
        max_episode_steps: int = None,
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
    # 1. ActionTransformWrapper:
    #    - Agent outputs actions in [-1, 1]^3.
    #    - This wrapper transforms them to CarRacing's native ranges ([~,ガス,ブレーキ]).
    #    - It defines self.action_space = Box([-1, 1]^3, ...) which SB3 will see.
    env = ActionTransformWrapper(env)

    # 2. ActionClipWrapper:
    #    - Clips actions received from the agent (which are in [-1, 1]^3 as per ActionTransformWrapper's space)
    #    - This ensures actions are strictly within the [-1, 1]^3 bounds before transformation by ActionTransformWrapper.
    #    - The ActionTransformWrapper then does its own clipping to the *underlying* environment's true bounds.
    env = ActionClipWrapper(env)  # This will clip to the action_space defined by ActionTransformWrapper

    # --- Observation Wrappers ---
    # 3. FrameStackWrapper: Stacks raw frames.
    env = FrameStackWrapper(env, frame_stack_num)

    # 4. LatentStateWrapper: Encodes stacked frames into quantized latent vectors using VQ-VAE.
    env = LatentStateWrapperVQ(env, vq_vae_model_instance, transform_function,
                               frame_stack_num, device_for_vae)

    # --- Reward Wrapper ---
    # 5. NormalizeReward: Normalizes rewards.
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
        transform_function=transform,  # Global transform from utils.py
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
