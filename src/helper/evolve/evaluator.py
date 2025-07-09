import pathlib
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import traceback

from sympy.printing.pytorch import torch

from src.utils import DEVICE, VQ_VAE_CHECKPOINT_FILENAME, make_env_sb3, ENV_NAME, NUM_STACK
from src.legacy.utils_legacy import transform
from src.vq_conv_vae import VQVAE

"""
OpenEvolve Evaluator for Reinforcement Learning programs
"""

SB3_MODEL_FILENAME = f"sb3_default_carracing-v3_best/best_model.zip"  # best
SB3_MODEL_PATH = pathlib.Path("../../../checkpoints") / SB3_MODEL_FILENAME


def evaluate_agent(agent_path: str, env_name: str, n_eval_episodes: int = 100) -> Dict[str, Any]:
    """
    Evaluates a trained SB3 agent.

    :param agent_path: Path to the saved agent .zip file.
    :param env_name: The name of the Gymnasium environment.
    :param n_eval_episodes: Number of episodes to run for evaluation.
    :return: A dictionary containing the evaluation metrics.
    """
    # --- Load VQ-VAE Model ---
    print(f"Loading VAE model to device: {DEVICE}")
    vq_vae_model = VQVAE().to(DEVICE)  # Ensure latent_dim is passed if constructor needs it
    try:
        vq_vae_model.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
        vq_vae_model.eval()
        print(f"Successfully loaded VAE: {VQ_VAE_CHECKPOINT_FILENAME}")
    except FileNotFoundError:
        print(f"ERROR: VAE checkpoint '{VQ_VAE_CHECKPOINT_FILENAME}' not found.")
        return
    except Exception as e:
        print(f"ERROR loading VAE: {e}")
        return

    # --- Create Environment using make_env_sb3 ---
    # make_env_sb3 handles all necessary wrappers including LatentStateWrapper and ActionTransformWrapper
    # It needs the VAE instance.
    # For playback, gamma for NormalizeReward wrapper doesn't strictly matter but use a sensible default.
    try:
        eval_env = make_env_sb3(
            env_id=ENV_NAME,
            vq_vae_model_instance=vq_vae_model,
            transform_function=transform,
            frame_stack_num=NUM_STACK,
            device_for_vae=DEVICE,
            gamma=0.99,  # Standard gamma, used by NormalizeReward
            render_mode="rgb_array",
            max_episode_steps=1000,  # Typical for CarRacing
        )
        print("Environment created successfully with make_env_sb3.")
    except Exception as e:
        print(f"Error creating environment with make_env_sb3: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load Trained SB3 PPO Agent ---
    print(f"Loading trained SB3 PPO agent from: {SB3_MODEL_PATH}")
    if not SB3_MODEL_PATH.exists():
        print(f"ERROR: SB3 PPO Model not found at {SB3_MODEL_PATH}")
        if hasattr(eval_env, 'close'): eval_env.close()
        return
    try:
        model = PPO.load(SB3_MODEL_PATH, device=DEVICE, env=eval_env)  # Provide eval_env for action/obs space checks
        print(f"Successfully loaded SB3 PPO agent. Agent device: {ppo_agent.device}")
    except Exception as e:
        print(f"ERROR loading SB3 PPO agent: {e}")
        if hasattr(eval_env, 'close'): eval_env.close()
        import traceback
        traceback.print_exc()
        return

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
    )

    # You can add more metrics here if needed, for example by inspecting the info dictionary
    # during a custom evaluation loop.

    metrics = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    }

    return metrics


def evaluate(program_path):
    """
    Evaluate the program by running it once and checking the sum of radii

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    print(f"Evaluating program at: {program_path}")
    try:
        evaluate_agent(agent_path=SB3_MODEL_PATH, env_name=ENV_NAME, n_eval_episodes=10)
        return {
            "metric": 0.1,
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        traceback.print_exc()
        return {
            "metric": 0.0,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluator.py <program_path>")
        sys.exit(1)

    program_path = sys.argv[1]
    metrics = evaluate(program_path)
    print(f"Evaluation metrics: {metrics}")
