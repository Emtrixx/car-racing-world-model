# train_ppo_sb3.py
import argparse
import random
import time
from typing import Callable

import torch
import optuna
from optuna.integration import MLflowCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.impala_cnn import CustomCNN
# Import from local modules
from src.utils import (
    DEVICE, ENV_NAME, NUM_STACK, _init_env_fn_sb3, CHECKPOINTS_DIR, LOGS_DIR, SB3_LOG_DIR
)

print(f"Using device for main script: {DEVICE}")
SB3_SAVE_DIR = CHECKPOINTS_DIR / "sb3_checkpoints"
SB3_SAVE_DIR.mkdir(parents=True, exist_ok=True)
SB3_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_config_sb3(name="default"):
    """
    Provides configurations for Stable Baselines3 PPO.
    Hyperparameters are adapted from the original PPO config.
    """
    configs = {
        "default": {
            # SB3 PPO Hyperparameters
            "policy": "CnnPolicy",
            "learning_rate": 45e-5,  # Can be a schedule
            "n_steps": 1024,  # Corresponds to STEPS_PER_BATCH (per environment)
            "batch_size": 64,  # PPO's minibatch size
            "n_epochs": 10,  # Corresponds to EPOCHS_PER_UPDATE
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.02,  # Corresponds to INITIAL_ENTROPY_COEF (fixed for now)
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.015,  # For early stopping in PPO updates
            "sde_sample_freq": -1,  # Set to -1 to disable SDE for standard PPO

            # Policy keyword arguments for MlpPolicy
            "policy_kwargs": dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=1024),
                net_arch=dict(pi=[256], vf=[256]),
                activation_fn=torch.nn.Tanh,
                log_std_init=-1.0,  # Matches custom Actor's initial log_std bias
                ortho_init=True,  # SB3 default, can be False if issues arise
            ),

            # Training parameters
            "total_timesteps": 1_000_000,
            "num_envs": 24,  # Number of parallel environments
            "save_freq": 50_000,  # Timesteps, not updates. (orig: 50 updates * 2048 steps/update = 102400 steps)
            # Let's use a step-based frequency for SB3 CheckpointCallback
            "eval_freq": 20480,  # Timesteps per eval environment
            "n_eval_episodes": 5,
            "seed": random.randint(0, 2 ** 31 - 1),  # generate a random seed for reproducibility

            # Environment parameters (passed to make_env_sb3 via _init_env_fn_sb3)
            "env_name_config": ENV_NAME,
            "num_stack_config": NUM_STACK,
            "gamma_config": 0.99,  # For NormalizeReward wrapper
            "max_episode_steps_config": 1000,
        }
    }
    configs["test"] = configs["default"].copy()
    configs["test"].update({
        "total_timesteps": 6_000,
        "n_steps": 128,
        "num_envs": 1,
        "save_freq": 5_000,
        "eval_freq": 2048,
        "learning_rate": 1e-4,
        "policy_kwargs": dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=512),  # Test with a smaller feature dim
            net_arch=dict(pi=[64], vf=[64]),  # Test specific architecture
            activation_fn=torch.nn.Tanh,  # Inherited or explicitly set
            log_std_init=-1.0,  # Inherited or explicitly set
            ortho_init=True,  # Inherited or explicitly set
        ),
    })
    return configs[name]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class OptunaPruningCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, eval_callback: EvalCallback):
        super().__init__()
        self.trial = trial
        self.eval_callback = eval_callback

    def _on_step(self) -> bool:
        if self.eval_callback.best_mean_reward is not None:
            self.trial.report(self.eval_callback.best_mean_reward, self.num_timesteps)
            if self.trial.should_prune():
                return False
        return True


def train_ppo_sb3(config_name: str, checkpoint_path: str = None, trial: optuna.Trial = None) -> float:
    """
    Train a PPO agent using Stable Baselines3 with the specified configuration.
    """
    print(f"Starting Stable Baselines3 PPO training with config: {config_name}...")
    config = get_config_sb3(config_name)
    start_time = time.time()

    if trial:
        # Hyperparameters to tune
        config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        config["n_steps"] = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        config["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
        config["n_epochs"] = trial.suggest_int("n_epochs", 5, 15)
        config["gamma"] = trial.suggest_float("gamma", 0.8, 0.9999)
        config["gae_lambda"] = trial.suggest_float("gae_lambda", 0.8, 0.99)
        config["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.3)
        config["ent_coef"] = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
        config["vf_coef"] = trial.suggest_float("vf_coef", 0.2, 0.9)
        config["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.3, 5.0)

        # For policy_kwargs
        features_dim = trial.suggest_categorical("features_dim", [256, 512, 1024])

        # Use tuples for list choices to make them hashable for the Optuna dashboard
        net_arch_choices_str = ("64", "128", "256", "64,64", "128,64")
        net_arch_pi_str = trial.suggest_categorical("net_arch_pi", net_arch_choices_str)
        net_arch_vf_str = trial.suggest_categorical("net_arch_vf", net_arch_choices_str)

        def parse_arch(arch_str: str) -> list[int]:
            if "," in arch_str:
                return [int(x) for x in arch_str.split(',')]
            else:
                return [int(arch_str)]

        net_arch_pi = parse_arch(net_arch_pi_str)
        net_arch_vf = parse_arch(net_arch_vf_str)

        config["policy_kwargs"]["features_extractor_kwargs"]["features_dim"] = features_dim
        config["policy_kwargs"]["net_arch"] = dict(pi=net_arch_pi, vf=net_arch_vf)
        # For optimization, run shorter trials
        config["total_timesteps"] = 250_000
        config["eval_freq"] = 10_000
        config["n_eval_episodes"] = 5
        config["num_envs"] = 16  # Use fewer envs for HPO to reduce overhead

    # generate seed and print it
    seed = config["seed"]
    print(f"Using seed: {seed}")
    set_random_seed(config["seed"])

    # Prepare parameters for environment creation
    env_params_for_creation = {
        "env_name_config": config["env_name_config"],
        "num_stack_config": config["num_stack_config"],
        "gamma_config": config["gamma_config"],
        "max_episode_steps_config": config["max_episode_steps_config"],
        # "render_mode": "human" if config["num_envs"] == 1 else None # Example for rendering
    }

    # Create vectorized environments
    if config["num_envs"] > 1:
        vec_env = SubprocVecEnv([
            lambda i=i: _init_env_fn_sb3(rank=i, seed=config["seed"], config_env_params=env_params_for_creation)
            for i in range(config["num_envs"])
        ])
    else:  # Use DummyVecEnv for single environment (easier debugging, but slower)
        vec_env = DummyVecEnv([
            lambda: _init_env_fn_sb3(rank=0, seed=config["seed"], config_env_params=env_params_for_creation)
        ])

    # Learning rate schedule
    lr_schedule = linear_schedule(config["learning_rate"])

    run_name = config_name
    if trial:
        run_name = f"{config_name}-trial-{trial.number}"

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config["save_freq"] // config["num_envs"], 1),  # Convert total steps to per-env steps
        save_path=str(SB3_SAVE_DIR / f"cnn_sb3_{run_name}_{ENV_NAME.lower()}"),
        name_prefix="ppo_model"
    )

    # Eval callback (optional, but good practice)
    # Create a separate evaluation environment (usually single, non-vectorized)
    eval_env_params = env_params_for_creation.copy()
    eval_env = DummyVecEnv([lambda: _init_env_fn_sb3(rank=config["num_envs"], seed=config["seed"] + 1000,
                                                     config_env_params=eval_env_params)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SB3_SAVE_DIR / f"cnn_sb3_{run_name}_{ENV_NAME.lower()}_best"),
        log_path=str(SB3_LOG_DIR / f"cnn_sb3_{run_name}_{ENV_NAME.lower()}_eval"),
        eval_freq=max(config["eval_freq"] // config["num_envs"], 1),
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False
    )

    callbacks = [eval_callback]
    if trial:
        pruning_callback = OptunaPruningCallback(trial, eval_callback)
        callbacks.append(pruning_callback)
    else:
        callbacks.append(checkpoint_callback)

    # PPO should be run on CPU
    # ppo_device = "cpu"
    ppo_device = DEVICE

    # Create PPO model
    model = PPO(
        policy=config["policy"],
        env=vec_env,
        learning_rate=lr_schedule,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        target_kl=config["target_kl"],
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=str(SB3_LOG_DIR / f"cnn_sb3_{run_name}_{ENV_NAME.lower()}"),
        verbose=0,
        seed=config["seed"],
        device=ppo_device  # SB3 will handle moving model to this device
    )

    if checkpoint_path:
        # Load a pre-trained model if provided
        try:
            print(f"Loading pre-trained model from {checkpoint_path}...")
            model.load(checkpoint_path, env=vec_env)
            print("Pre-trained model loaded successfully.")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            import traceback
            traceback.print_exc()

    print(f"PPO Model Device: {model.device}")
    print(f"Observation Space: {model.observation_space}")
    print(f"Action Space: {model.action_space}")

    # Train the agent
    print(f"Training for {config['total_timesteps']} total timesteps...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            progress_bar=True if not trial else False
        )
    except optuna.exceptions.TrialPruned as e:
        print(f"Trial pruned: {e}")
        vec_env.close()
        eval_env.close()
        raise e
    except Exception as e:
        print(f"Error during model.learn: {e}")
        import traceback
        traceback.print_exc()
        vec_env.close()
        eval_env.close()
        return -1.0  # Return a bad value for failed trials
    finally:
        if not trial:
            # Save the final model only on regular runs
            final_model_path = SB3_SAVE_DIR / f"cnn_sb3_{config_name}_{ENV_NAME.lower()}_final.zip"
            model.save(final_model_path)
            print(f"Final model saved to {final_model_path}")

        vec_env.close()  # Important to close vectorized environments
        eval_env.close()

    total_time = time.time() - start_time
    print(f"Training finished. Total training time: {total_time:.2f} seconds")
    if not trial:
        print(f"Models and logs saved in: {SB3_SAVE_DIR} and {SB3_LOG_DIR}")

    best_reward = eval_callback.best_mean_reward
    print(f"Best mean reward: {best_reward}")
    return best_reward


def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna optimization.
    """
    return train_ppo_sb3(config_name="default", trial=trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent using Stable Baselines3.")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the configuration to use (e.g., 'default', 'test')."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained model checkpoint to load before training. If provided, "
             "the model will be loaded and training will continue from that point."
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization with Optuna."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=500,
        help="Number of trials for Optuna optimization."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="ppo_optimization",
        help="Name for the Optuna study."
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="postgresql://optuna:optuna@db:5432/optuna",
        help="Database storage for Optuna study."
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="logs/wm_mlflow",
        help="MLflow tracking URI."
    )
    args = parser.parse_args()

    if args.optimize:
        print("Starting hyperparameter optimization with Optuna...")

        # Increase the timeout for the SQL connection to prevent "database is locked" errors
        storage = optuna.storages.RDBStorage(
            url=args.storage,
            engine_kwargs={"connect_args": {"timeout": 30}},  # Set timeout to 30 seconds
        )

        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        mlflow_callback = MLflowCallback(
            tracking_uri=args.mlflow_tracking_uri,
            create_experiment=True,
            metric_name="mean_reward",
        )

        try:
            study.optimize(objective, n_trials=args.n_trials, timeout=3600 * 6, callbacks=[mlflow_callback])
        except KeyboardInterrupt:
            print("Optimization stopped manually.")

        print("Optimization finished.")
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    else:
        train_ppo_sb3(args.config, checkpoint_path=args.checkpoint)
