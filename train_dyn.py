import argparse
import os
import pathlib
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # For logging

from conv_vae import ConvVAE
from utils import (
    DEVICE, transform, LATENT_DIM, ACTION_DIM, NUM_STACK,
    VAE_CHECKPOINT_FILENAME,
    WM_CHECKPOINT_FILENAME_GRU,
    ENV_NAME,
    FrameStackWrapper, ActionClipWrapper, LatentStateWrapper, ActionTransformWrapper,
    preprocess_and_encode
)

# from stable_baselines3.common.vec_env import DummyVecEnv # Not used directly

SB3_PPO_DEFAULT_PATH = "checkpoints/sb3_default_carracing-v3_best/best_model.zip"


# --- Helper for printing section headers ---
def print_section_header(title):
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70)


def print_subsection_header(title):
    print("\n" + "-" * 60)
    print(f"{title.center(60)}")
    print("-" * 60)


# --- GRU World Model for Continuous Latent States ---
class ContinuousGRUWorldModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=256, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gru_num_layers = num_layers

        self.gru = nn.GRU(latent_dim + action_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out_latent = nn.Linear(hidden_dim, latent_dim)
        self.fc_out_reward = nn.Linear(hidden_dim, 1)
        self.fc_out_done_logit = nn.Linear(hidden_dim, 1)

    def forward(self, z_t, a_t, h_prev=None):
        if z_t.ndim == 1: z_t = z_t.unsqueeze(0)
        if a_t.ndim == 1: a_t = a_t.unsqueeze(0)
        if z_t.shape[0] == a_t.shape[0]:
            za_t = torch.cat([z_t, a_t], dim=1)
            if za_t.ndim == 2:
                za_t_seq = za_t.unsqueeze(1)
            else:
                za_t_seq = za_t.unsqueeze(0)
        else:
            raise ValueError(f"Shape mismatch or unexpected ndim for z_t ({z_t.shape}) or a_t ({a_t.shape})")

        gru_out, h_next = self.gru(za_t_seq, h_prev)
        last_gru_out = gru_out[:, -1, :]
        next_z_pred = self.fc_out_latent(last_gru_out)
        reward_pred = self.fc_out_reward(last_gru_out)
        done_pred_logit = self.fc_out_done_logit(last_gru_out)
        return next_z_pred, reward_pred, done_pred_logit, h_next

    def predict_sequence(self, z_start, actions_sequence, h_initial=None):
        batch_size, seq_len, _ = actions_sequence.shape
        current_z = z_start
        hidden_state = h_initial
        predicted_latents, predicted_rewards, predicted_done_logits = [], [], []
        for t in range(seq_len):
            action_t = actions_sequence[:, t, :]
            next_z_pred_wm, reward_pred_wm, done_pred_logit_wm, hidden_state = self.forward(current_z, action_t,
                                                                                            hidden_state)
            predicted_latents.append(next_z_pred_wm)
            predicted_rewards.append(reward_pred_wm)
            predicted_done_logits.append(done_pred_logit_wm)
            current_z = next_z_pred_wm
        return torch.stack(predicted_latents, dim=1), \
            torch.stack(predicted_rewards, dim=1), \
            torch.stack(predicted_done_logits, dim=1)


# --- Model Loading and Saving Functions ---
def load_vae_model(vae_path, device):
    model = ConvVAE(img_channels=3, latent_dim=LATENT_DIM).to(device)
    try:
        model.load_state_dict(torch.load(vae_path, map_location=device))
        model.eval();
        print(f"VAE model loaded from {vae_path}")
        return model
    except Exception as e:
        print(f"ERROR loading VAE from {vae_path}: {e}");
        return None


def load_sb3_ppo_model(model_path, device_str="cpu"):
    try:
        ppo_agent = PPO.load(model_path, device=device_str)
        print(f"SB3 PPO model loaded from {model_path} device '{ppo_agent.device}'")
        return ppo_agent
    except Exception as e:
        print(f"ERROR loading SB3 PPO from {model_path}: {e}");
        return None


def initialize_world_model(latent_dim, action_dim, hidden_dim, num_layers, device, checkpoint_path=None):
    world_model = ContinuousGRUWorldModel(latent_dim, action_dim, hidden_dim, num_layers).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            world_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            world_model.eval();
            print(f"World Model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading WM from {checkpoint_path}: {e}. New model used.")
    elif checkpoint_path:
        print(f"Warning: WM checkpoint {checkpoint_path} not found. New model used.")
    else:
        print("Initializing new World Model (no checkpoint path).")
    return world_model


def save_sb3_ppo_model(ppo_agent, save_dir, base_filename, iteration):
    path = pathlib.Path(save_dir) / f"{base_filename}_iter{iteration}.zip"
    ppo_agent.save(path)
    print(f"  PPO model saved to {path}")


def save_torch_model(model, save_dir, base_filename, iteration):
    path = pathlib.Path(save_dir) / f"{base_filename}_iter{iteration}.pth"
    torch.save(model.state_dict(), path)
    print(f"  Torch model ({base_filename}) saved to {path}")


# --- Environment Initialization ---
def make_conv_vae_env(env_id, seed, vae_model, num_frame_stack, device_for_vae, render_mode, max_ep_steps,
                      gamma_normalize_reward=0.99):
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_ep_steps)
    env.action_space.seed(seed)
    env.reset(seed=seed)
    env = ActionTransformWrapper(env)
    env = ActionClipWrapper(env)
    env = FrameStackWrapper(env, num_stack=num_frame_stack)
    env = LatentStateWrapper(env, vae_model, transform, LATENT_DIM, num_frame_stack, device_for_vae)
    if gamma_normalize_reward is not None:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma_normalize_reward)
    return env


# --- World Model Dataset ---
class WorldModelDataset(Dataset):
    def __init__(self, buffer):
        self.z_t_list = [torch.as_tensor(s[0], dtype=torch.float32) for s in buffer]
        self.a_t_list = [torch.as_tensor(s[1], dtype=torch.float32) for s in buffer]
        self.r_tp1_list = [torch.as_tensor(s[2], dtype=torch.float32) for s in buffer]
        self.z_tp1_list = [torch.as_tensor(s[3], dtype=torch.float32) for s in buffer]
        self.d_tp1_list = [torch.as_tensor(s[4], dtype=torch.float32) for s in buffer]

    def __len__(self): return len(self.z_t_list)

    def __getitem__(self, idx):
        return (self.z_t_list[idx], self.a_t_list[idx], self.r_tp1_list[idx],
                self.z_tp1_list[idx], self.d_tp1_list[idx])


# --- Data Collection & Training Functions ---
def calculate_average_reward(ppo_buffer):
    if not ppo_buffer: return 0.0
    return np.mean([sample[2] for sample in ppo_buffer])  # sample[2] is reward


def collect_real_samples_sb3(env, ppo_agent, vae_model, num_episodes, transform_fn, device_for_vae_encoding):
    ppo_buffer, wm_buffer = [], []
    total_steps_collected = 0
    vae_model.eval()
    for episode_idx in range(num_episodes):
        current_stacked_latent_obs_numpy, info = env.reset()
        frame_stack_wrapper = env
        while not isinstance(frame_stack_wrapper, FrameStackWrapper) and hasattr(frame_stack_wrapper, 'env'):
            frame_stack_wrapper = frame_stack_wrapper.env
        if not isinstance(frame_stack_wrapper, FrameStackWrapper):
            raise ValueError("Could not find FrameStackWrapper for raw frame access.")
        most_recent_raw_frame_t = frame_stack_wrapper.frames[-1]
        with torch.no_grad():
            single_z_t_tensor = preprocess_and_encode(
                most_recent_raw_frame_t, transform_fn, vae_model, device_for_vae_encoding)
        episode_steps, episode_reward = 0, 0
        done, truncated = False, False
        while not done and not truncated:
            action_from_ppo, _ = ppo_agent.predict(current_stacked_latent_obs_numpy, deterministic=False)
            next_stacked_latent_obs_numpy, reward, done, truncated, info = env.step(action_from_ppo)
            most_recent_raw_frame_t_plus_1 = frame_stack_wrapper.frames[-1]
            with torch.no_grad():
                single_z_t_plus_1_tensor = preprocess_and_encode(
                    most_recent_raw_frame_t_plus_1, transform_fn, vae_model, device_for_vae_encoding)
            ppo_buffer.append((current_stacked_latent_obs_numpy, action_from_ppo, reward,
                               next_stacked_latent_obs_numpy, float(done or truncated)))
            wm_buffer.append((single_z_t_tensor.cpu().numpy(), action_from_ppo, reward,
                              single_z_t_plus_1_tensor.cpu().numpy(), float(done or truncated)))
            current_stacked_latent_obs_numpy = next_stacked_latent_obs_numpy
            single_z_t_tensor = single_z_t_plus_1_tensor
            episode_reward += reward;
            episode_steps += 1;
            total_steps_collected += 1
        print(f"    Episode {episode_idx + 1}/{num_episodes}: Steps={episode_steps}, Reward={episode_reward:.2f}")
    print(f"    Total steps collected this phase: {total_steps_collected}")
    return ppo_buffer, wm_buffer


def train_ppo_on_buffer(ppo_agent, ppo_samples_buffer, args, device, is_real_data: bool, writer: SummaryWriter,
                        dyna_iter: int):
    if not ppo_samples_buffer:
        print(f"  PPO sample buffer ({'real' if is_real_data else 'synthetic'}) is empty. Skipping training.")
        return
    if is_real_data:
        n_epochs, batch_size, ds_name = args.ppo_update_epochs_real, args.ppo_batch_size_real, "real"
        log_tag_prefix = "ppo_train_real"
    else:
        n_epochs, batch_size, ds_name = args.ppo_update_epochs_synthetic, args.ppo_batch_size_synthetic, "synthetic"
        log_tag_prefix = "ppo_train_synthetic"
    print(f"    Training PPO on {ds_name} data: Epochs={n_epochs}, BatchSize={batch_size}")
    avg_reward_buffer = calculate_average_reward(ppo_samples_buffer)
    writer.add_scalar(f"{log_tag_prefix}/avg_buffer_reward", avg_reward_buffer, dyna_iter)

    observations_np = np.array([s[0] for s in ppo_samples_buffer])
    actions_np = np.array([s[1] for s in ppo_samples_buffer])
    rewards_np = np.array([s[2] for s in ppo_samples_buffer])
    next_observations_np = np.array([s[3] for s in ppo_samples_buffer])
    dones_np = np.array([s[4] for s in ppo_samples_buffer])
    episode_starts_np = np.zeros_like(dones_np, dtype=bool);
    episode_starts_np[0] = True
    for i in range(1, len(dones_np)): episode_starts_np[i] = dones_np[i - 1]
    ppo_agent.policy.to(device);
    ppo_agent.policy.eval()
    values_list, log_probs_list = [], []
    with torch.no_grad():
        for obs_np, action_np in zip(observations_np, actions_np):
            obs_tensor = torch.as_tensor(obs_np).float().to(device).unsqueeze(0)
            action_tensor = torch.as_tensor(action_np).float().to(device).unsqueeze(0)
            value_tensor = ppo_agent.policy.predict_values(obs_tensor)
            _, log_prob_tensor, _ = ppo_agent.policy.evaluate_actions(obs_tensor, action_tensor)
            values_list.append(value_tensor.squeeze().cpu().numpy())
            log_probs_list.append(log_prob_tensor.cpu().numpy().flatten())
    values_np, log_probs_np = np.array(values_list), np.array(log_probs_list).flatten()
    rollout_buffer = ppo_agent.rollout_buffer;
    rollout_buffer.reset()
    # Handle buffer size warning more gracefully or ensure PPO n_steps is large enough
    if rollout_buffer.buffer_size < len(observations_np):
        print(f"    Warning: PPO agent's rollout_buffer size ({rollout_buffer.buffer_size}) is smaller than "
              f"collected samples ({len(observations_np)}). This might truncate data. "
              f"Consider increasing PPO agent's n_steps parameter if this is unintended.")
    for i in range(len(observations_np)):
        rollout_buffer.add(observations_np[i], actions_np[i], rewards_np[i],
                           episode_starts_np[i], values_np[i], log_probs_np[i])
    last_next_obs_tensor = torch.as_tensor(next_observations_np[-1]).float().to(device).unsqueeze(0)
    with torch.no_grad():
        last_value_tensor = ppo_agent.policy.predict_values(last_next_obs_tensor)
    rollout_buffer.compute_returns_and_advantage(last_values=last_value_tensor.cpu(), dones=np.array([dones_np[-1]]))
    ppo_agent.policy.train()
    original_n_epochs, original_batch_size = ppo_agent.n_epochs, ppo_agent.batch_size
    ppo_agent.n_epochs, ppo_agent.batch_size = n_epochs, batch_size
    try:
        ppo_agent.train()
        print(f"    PPO training on {ds_name} data performed for {n_epochs} epochs.")
    except Exception as e:
        print(f"    ERROR during PPO training on {ds_name} data: {e}");
        import traceback;
        traceback.print_exc()
    finally:
        ppo_agent.n_epochs, ppo_agent.batch_size = original_n_epochs, original_batch_size
        ppo_agent.policy.eval()


def train_world_model_on_samples(world_model, wm_samples_buffer, args, device, writer: SummaryWriter, dyna_iter: int):
    if not wm_samples_buffer: print("  World Model sample buffer is empty. Skipping WM training."); return
    dataset = WorldModelDataset(wm_samples_buffer)
    # Added num_workers and pin_memory for potentially faster data loading
    dataloader = DataLoader(dataset, batch_size=args.wm_batch_size_real, shuffle=True,
                            num_workers=os.cpu_count() // 2 or 1,
                            pin_memory=True if device != torch.device("cpu") else False)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=args.wm_lr_real)
    z_loss_fn, r_loss_fn, d_loss_fn = nn.MSELoss(), nn.MSELoss(), nn.BCEWithLogitsLoss()
    world_model.train()
    # Initialize accumulators for iteration-level logging
    iter_avg_loss_z, iter_avg_loss_r, iter_avg_loss_d, iter_avg_total_loss = 0, 0, 0, 0

    for epoch in range(args.wm_epochs_real):
        epoch_loss_z, epoch_loss_r, epoch_loss_d, epoch_total_loss, num_batches = 0, 0, 0, 0, 0
        for z_t, a_t, r_target, z_target, d_target in dataloader:
            z_t, a_t, r_target, z_target, d_target = [x.to(device) for x in [z_t, a_t, r_target, z_target, d_target]]
            next_z_pred, reward_pred, done_pred_logit, _ = world_model(z_t, a_t)
            loss_z = z_loss_fn(next_z_pred, z_target)
            loss_r = r_loss_fn(reward_pred.squeeze(1), r_target)
            loss_d = d_loss_fn(done_pred_logit.squeeze(1), d_target)
            total_batch_loss = loss_z + args.wm_beta_real * (loss_r + loss_d)
            optimizer.zero_grad();
            total_batch_loss.backward();
            optimizer.step()
            epoch_loss_z += loss_z.item();
            epoch_loss_r += loss_r.item();
            epoch_loss_d += loss_d.item()
            epoch_total_loss += total_batch_loss.item();
            num_batches += 1

        # Calculate averages for the current epoch
        avg_loss_z_epoch = epoch_loss_z / num_batches
        avg_loss_r_epoch = epoch_loss_r / num_batches
        avg_loss_d_epoch = epoch_loss_d / num_batches
        avg_total_loss_epoch = epoch_total_loss / num_batches

        print(f"    WM Epoch {epoch + 1}/{args.wm_epochs_real}: Avg Loss={avg_total_loss_epoch:.4f} "
              f"[Z: {avg_loss_z_epoch:.4f}, R: {avg_loss_r_epoch:.4f}, D: {avg_loss_d_epoch:.4f}]")
        if writer:
            # Log per epoch, using a combined step for uniqueness across dyna_iters
            global_epoch_step = dyna_iter * args.wm_epochs_real + epoch
            writer.add_scalar("wm_train_epoch/total_loss", avg_total_loss_epoch, global_epoch_step)
            writer.add_scalar("wm_train_epoch/z_loss", avg_loss_z_epoch, global_epoch_step)
            writer.add_scalar("wm_train_epoch/r_loss", avg_loss_r_epoch, global_epoch_step)
            writer.add_scalar("wm_train_epoch/d_loss", avg_loss_d_epoch, global_epoch_step)

        # Accumulate last epoch's averages for iteration-level logging
        if epoch == args.wm_epochs_real - 1:
            iter_avg_loss_z, iter_avg_loss_r, iter_avg_loss_d, iter_avg_total_loss = \
                avg_loss_z_epoch, avg_loss_r_epoch, avg_loss_d_epoch, avg_total_loss_epoch

    world_model.eval()
    if writer:  # Log average over the last epoch for this dyna_iter
        writer.add_scalar("wm_train_iter/avg_total_loss", iter_avg_total_loss, dyna_iter)
        writer.add_scalar("wm_train_iter/avg_z_loss", iter_avg_loss_z, dyna_iter)
        writer.add_scalar("wm_train_iter/avg_r_loss", iter_avg_loss_r, dyna_iter)
        writer.add_scalar("wm_train_iter/avg_d_loss", iter_avg_loss_d, dyna_iter)


def collect_synthetic_samples(world_model, ppo_agent, initial_real_z_states, args, device):
    if not initial_real_z_states: print("  No real initial states for dreaming. Skipping."); return []
    synthetic_ppo_samples = []
    world_model.to(device).eval();
    ppo_agent.policy.to(device).eval()

    for i in range(args.num_synthetic_sequences):
        idx = random.randint(0, len(initial_real_z_states) - 1)
        current_single_z = torch.as_tensor(initial_real_z_states[idx], dtype=torch.float32).to(device)
        current_ppo_latent_stack = deque(maxlen=args.num_stack)
        for _ in range(args.num_stack): current_ppo_latent_stack.append(current_single_z.clone())
        stacked_obs_for_ppo = torch.cat(list(current_ppo_latent_stack), dim=0)
        h_t = torch.zeros(world_model.gru_num_layers, 1, world_model.hidden_dim, device=device)

        for _ in range(args.synthetic_horizon):
            action_np, _ = ppo_agent.predict(stacked_obs_for_ppo.cpu().numpy(), deterministic=False)
            action_tensor = torch.as_tensor(action_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                next_single_z_pred, reward_pred, done_pred_logit, h_next = world_model(
                    current_single_z.unsqueeze(0), action_tensor.unsqueeze(0), h_t)
            next_single_z_pred = next_single_z_pred.squeeze(0)
            reward_pred = reward_pred.squeeze(0).squeeze(0)
            done_pred_logit = done_pred_logit.squeeze(0).squeeze(0)
            done_flag = (torch.sigmoid(done_pred_logit) > args.dream_done_threshold).float()
            current_ppo_latent_stack.append(next_single_z_pred.clone())
            next_stacked_obs_for_ppo = torch.cat(list(current_ppo_latent_stack), dim=0)
            synthetic_ppo_samples.append((stacked_obs_for_ppo.cpu().numpy(), action_np,
                                          reward_pred.item(), next_stacked_obs_for_ppo.cpu().numpy(),
                                          done_flag.item()))
            stacked_obs_for_ppo, current_single_z, h_t = next_stacked_obs_for_ppo, next_single_z_pred, h_next
            if done_flag.item() == 1.0: break
        if (i + 1) % (args.num_synthetic_sequences // 10 or 1) == 0:
            print(f"    Generated {i + 1}/{args.num_synthetic_sequences} synthetic sequences.")
    return synthetic_ppo_samples


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Train DYNA-style algorithm with SB3 PPO and a GRU World Model.')
    # Group: Environment and Model Paths
    group_paths = parser.add_argument_group('Paths')
    group_paths.add_argument('--env_name', type=str, default=ENV_NAME,
                             help='Gym environment name. Default: %(default)s')
    group_paths.add_argument('--vae_path', type=str, default=VAE_CHECKPOINT_FILENAME,
                             help='Path to ConvVAE checkpoint. Default: %(default)s')
    group_paths.add_argument('--sb3_ppo_path', type=str, default=SB3_PPO_DEFAULT_PATH,
                             help='Path to SB3 PPO agent ZIP checkpoint. Default: %(default)s')
    group_paths.add_argument('--wm_path', type=str, default=WM_CHECKPOINT_FILENAME_GRU,
                             help='Path to GRU World Model checkpoint (compatible with ContinuousGRUWorldModel). Default: %(default)s')

    # Group: Core Model Hyperparameters
    group_core_model = parser.add_argument_group('Core Model Hyperparameters')
    group_core_model.add_argument('--latent_dim', type=int, default=LATENT_DIM,
                                  help='Dimension of VAE latent space. Default: %(default)s')
    group_core_model.add_argument('--num_stack', type=int, default=NUM_STACK,
                                  help='Number of frames/latents to stack for PPO agent observation. Default: %(default)s')
    group_core_model.add_argument('--wm_hidden_dim', type=int, default=256,
                                  help='Hidden dimension for World Model GRU. Default: %(default)s')
    group_core_model.add_argument('--wm_gru_layers', type=int, default=1,
                                  help='Number of GRU layers in World Model. Default: %(default)s')
    group_core_model.add_argument('--seed', type=int, default=42,
                                  help='Random seed for reproducibility. Default: %(default)s')

    # Group: Data Collection (Real Environment)
    group_collect_real = parser.add_argument_group('Real Data Collection')
    group_collect_real.add_argument('--collect_episodes', type=int, default=10,
                                    help='Episodes for initial real data collection per Dyna iteration. Default: %(default)s')
    group_collect_real.add_argument('--max_steps_per_episode_collect', type=int, default=300,
                                    help='Max steps per episode during real data collection. Default: %(default)s')
    group_collect_real.add_argument('--render_collection', action='store_true',
                                    help='Render environment during real data collection.')
    group_collect_real.add_argument('--wm_collect_episodes_after_ppo_update', type=int, default=None,
                                    help='Episodes for WM data collection after PPO update. If None, defaults to --collect_episodes. Default: %(default)s')

    # Group: PPO Training (on Real Data)
    group_ppo_real = parser.add_argument_group('PPO Training on Real Data')
    group_ppo_real.add_argument('--ppo_update_epochs_real', type=int, default=4,
                                help='Epochs for PPO update on real samples. Default: %(default)s')
    group_ppo_real.add_argument('--ppo_batch_size_real', type=int, default=32,
                                help='Minibatch size for PPO on real samples. Default: %(default)s')

    # Group: World Model Training (on Real Data)
    group_wm_real = parser.add_argument_group('World Model Training on Real Data')
    group_wm_real.add_argument('--wm_epochs_real', type=int, default=5,
                               help='Epochs for WM training on real samples. Default: %(default)s')
    group_wm_real.add_argument('--wm_batch_size_real', type=int, default=32,
                               help='Minibatch size for WM training. Default: %(default)s')
    group_wm_real.add_argument('--wm_lr_real', type=float, default=1e-4,
                               help='Learning rate for WM training. Default: %(default)s')
    group_wm_real.add_argument('--wm_beta_real', type=float, default=0.5,
                               help='Weighting for reward/done losses in WM training, relative to state loss. Default: %(default)s')

    # Group: Synthetic Data Generation (Dreaming)
    group_dream = parser.add_argument_group('Synthetic Data Generation (Dreaming)')
    group_dream.add_argument('--num_synthetic_sequences', type=int, default=200,
                             help='Number of synthetic sequences per Dyna iter. Default: %(default)s')
    group_dream.add_argument('--synthetic_horizon', type=int, default=30,
                             help='Max length of each synthetic sequence (dream horizon). Default: %(default)s')
    group_dream.add_argument('--dream_done_threshold', type=float, default=0.5,
                             help='Sigmoid threshold for predicted done logit to terminate a dream sequence. Default: %(default)s')

    # Group: PPO Training (on Synthetic Data)
    group_ppo_synth = parser.add_argument_group('PPO Training on Synthetic Data')
    group_ppo_synth.add_argument('--ppo_update_epochs_synthetic', type=int, default=2,
                                 help='Epochs for PPO update on synthetic samples. Default: %(default)s')
    group_ppo_synth.add_argument('--ppo_batch_size_synthetic', type=int, default=32,
                                 help='Minibatch size for PPO on synthetic samples. Default: %(default)s')

    # Group: Main Dyna Loop Configuration
    group_dyna = parser.add_argument_group('Dyna Algorithm Configuration')
    group_dyna.add_argument('--dyna_iterations', type=int, default=5,
                            help='Number of Dyna iterations to run. Default: %(default)s')

    # Group: Logging and Model Saving
    group_log_save = parser.add_argument_group('Logging and Saving')
    group_log_save.add_argument('--log_dir', type=str, default='logs/dyna_run',
                                help='Directory for TensorBoard logs. Default: %(default)s')
    group_log_save.add_argument('--save_model_dir', type=str, default='models/dyna_run',
                                help='Directory for saving models. Default: %(default)s')
    group_log_save.add_argument('--save_model_freq', type=int, default=1,
                                help='Frequency (in Dyna iterations) to save models. Set to 0 to disable. Default: %(default)s')

    args = parser.parse_args()

    print_section_header("Initial Model Loading and Setup")
    print(f"Using PyTorch device: {DEVICE}")
    print(f"Script arguments: {args}")
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    if args.save_model_freq > 0:
        pathlib.Path(args.save_model_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    vae_model = load_vae_model(args.vae_path, DEVICE)
    if vae_model is None: writer.close(); return
    ppo_agent = load_sb3_ppo_model(args.sb3_ppo_path, device_str="cpu")
    if ppo_agent is None: writer.close(); return
    world_model = initialize_world_model(
        args.latent_dim, ACTION_DIM, args.wm_hidden_dim, args.wm_gru_layers, DEVICE, args.wm_path)
    print("VAE, SB3 PPO Agent, and World Model initialized/loaded.\n")

    all_wm_real_samples = []

    for dyna_iter in range(1, args.dyna_iterations + 1):
        print_section_header(f"Dyna Iteration {dyna_iter}/{args.dyna_iterations}")

        print_subsection_header("Step 3a: Initial Real Environment Data Collection")
        real_env_iter = make_conv_vae_env(
            args.env_name, args.seed + dyna_iter - 1, vae_model, args.num_stack,
            DEVICE, args.render_collection, args.max_steps_per_episode_collect)
        current_iter_ppo_samples, current_iter_wm_samples = collect_real_samples_sb3(
            real_env_iter, ppo_agent, vae_model, args.collect_episodes, transform, DEVICE)
        writer.add_scalar("real_samples/num_ppo_samples", len(current_iter_ppo_samples), dyna_iter)
        writer.add_scalar("real_samples/num_wm_samples", len(current_iter_wm_samples), dyna_iter)
        writer.add_scalar("real_samples/avg_reward_initial_collect", calculate_average_reward(current_iter_ppo_samples),
                          dyna_iter)
        print(f"  Collected {len(current_iter_ppo_samples)} PPO samples and {len(current_iter_wm_samples)} WM samples.")
        real_env_iter.close()

        if current_iter_ppo_samples:
            print_subsection_header("Step 3b: Training PPO on Real Samples")
            train_ppo_on_buffer(ppo_agent, current_iter_ppo_samples, args, torch.device(ppo_agent.device), True, writer,
                                dyna_iter)
        else:
            print("  No PPO real samples this iteration, skipping PPO training on real data.")

        print_subsection_header("Step 3c: WM Data Collection with Updated PPO Policy")
        num_wm_episodes_updated = args.wm_collect_episodes_after_ppo_update if args.wm_collect_episodes_after_ppo_update is not None else args.collect_episodes
        if num_wm_episodes_updated > 0:
            real_env_wm_iter = make_conv_vae_env(
                args.env_name, args.seed + dyna_iter - 1 + 1000, vae_model, args.num_stack,
                DEVICE, args.render_collection, args.max_steps_per_episode_collect)
            ppo_samples_for_wm_coll, additional_wm_samples = collect_real_samples_sb3(
                real_env_wm_iter, ppo_agent, vae_model, num_wm_episodes_updated, transform, DEVICE)
            writer.add_scalar("wm_samples_updated_ppo/num_samples", len(additional_wm_samples), dyna_iter)
            writer.add_scalar("wm_samples_updated_ppo/avg_reward", calculate_average_reward(ppo_samples_for_wm_coll),
                              dyna_iter)
            print(f"  Collected {len(additional_wm_samples)} additional WM samples with updated PPO.")
            current_iter_wm_samples.extend(additional_wm_samples)
            real_env_wm_iter.close()
        else:
            print("  Skipping WM data collection with updated PPO (episodes set to 0).")

        if current_iter_wm_samples:
            all_wm_real_samples.extend(current_iter_wm_samples)
            writer.add_scalar("wm_samples_total/accumulated_real_wm_samples", len(all_wm_real_samples), dyna_iter)
            print(f"  Total accumulated WM real samples: {len(all_wm_real_samples)}")

        if all_wm_real_samples:
            print_subsection_header("Step 3d: Training World Model on All Accumulated Real Samples")
            train_world_model_on_samples(world_model, all_wm_real_samples, args, DEVICE, writer, dyna_iter)
        else:
            print("  No WM samples to train World Model. Skipping.")

        dream_ppo_samples = []
        if world_model and all_wm_real_samples:  # Check if all_wm_real_samples is not empty
            print_subsection_header("Step 3e: Synthetic Sample Collection (Dreaming)")
            dream_ppo_samples = collect_synthetic_samples(world_model, ppo_agent, [s[0] for s in all_wm_real_samples],
                                                          args, DEVICE)
            writer.add_scalar("synthetic_samples/num_dream_ppo_samples", len(dream_ppo_samples), dyna_iter)
            writer.add_scalar("synthetic_samples/avg_reward_dream_buffer", calculate_average_reward(dream_ppo_samples),
                              dyna_iter)
            print(f"  Collected {len(dream_ppo_samples)} synthetic PPO samples from dreams.")
        else:
            print("  Skipping dreaming: World Model not trained or no initial real states for dreaming.")

        if dream_ppo_samples:
            print_subsection_header("Step 3f: Training PPO on Synthetic Samples")
            train_ppo_on_buffer(ppo_agent, dream_ppo_samples, args, torch.device(ppo_agent.device), False, writer,
                                dyna_iter)
        else:
            print("  No synthetic samples to train PPO. Skipping.")

        if args.save_model_freq > 0 and dyna_iter % args.save_model_freq == 0:
            print_subsection_header(f"Saving Models at Dyna Iteration {dyna_iter}")
            save_sb3_ppo_model(ppo_agent, args.save_model_dir, "ppo_agent_dyna", dyna_iter)
            save_torch_model(world_model, args.save_model_dir, "world_model_dyna", dyna_iter)
        print_section_header(f"Dyna Iteration {dyna_iter} Complete")

    print("\nAll Dyna iterations complete.")
    writer.close()
    print(f"TensorBoard logs saved to: {args.log_dir}")
    if args.save_model_freq > 0: print(f"Models saved in: {args.save_model_dir}")


if __name__ == '__main__':
    main()
