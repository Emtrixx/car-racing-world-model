import os
import subprocess
import multiprocessing
import argparse
import time

import optuna

# =============================================================================
#  CONFIGURATION
# =============================================================================
CONFIGS = {
    "ppo": {
        "description": "Hyperparameter sweep for PPO.",
        "num_workers": 16,
        "study_name": "ppo_optimization",
        "command_args": [
            "python",
            "-m",
            "src.train_ppo_sb3",
            "--optimize",
        ]
    },
    "transformer_wm": {
        "description": "Hyperparameter sweep for the Transformer World Model.",
        "num_workers": 8,
        "study_name": "transformer_wm_optimization",
        "command_args": [
            "python",
            "-m",
            "src.train_transformer_world_model",
            "--optimize",
            "--load-data-from=./data/transformer_world_model_data",
        ]
    },
}


# =============================================================================

def run_worker(worker_id_and_command):
    """Function to be executed by each worker process."""
    worker_id, command, delay = worker_id_and_command
    time.sleep(delay)  # delay for staggered starts
    print(f"Starting worker {worker_id}...")
    try:
        # Each worker will independently run the training script.
        subprocess.run(command, check=True)
        print(f"Worker {worker_id} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Worker {worker_id} failed with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in worker {worker_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A parallel worker launcher for optimization tasks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=CONFIGS.keys(),
        help="The name of the configuration to run. Available choices are:\n" +
             "\n".join([f"  - {name}: {conf.get('description', 'No description')} " for name, conf in CONFIGS.items()])
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers to start. Overrides the default in the selected config."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study. If not provided, uses the default from the selected config."
    )

    args = parser.parse_args()

    # --- Select and prepare the configuration ---
    selected_config = CONFIGS[args.config]
    command_to_run = selected_config["command_args"]
    num_workers = args.num_workers if args.num_workers is not None else selected_config["num_workers"]
    study_name = args.study_name if args.study_name is not None else selected_config["study_name"]

    # --- Define study parameters ---
    # Use PostgreSQL connection string from environment variables
    user = os.getenv("POSTGRES_USER", "optuna")
    password = os.getenv("POSTGRES_PASSWORD", "optuna")
    host = os.getenv("POSTGRES_HOST", "db")
    dbname = os.getenv("POSTGRES_DB", "optuna")
    storage_path = f"postgresql://{user}:{password}@{host}:5432/{dbname}"

    mlflow_tracking_uri = "logs/wm_mlflow"

    # --- Create the study before starting workers to prevent race conditions ---
    if args.config == "ppo":
        print(f"Ensuring Optuna study '{study_name}' exists in {storage_path}...")
        # Add a timeout to handle initial DB connection
        storage = optuna.storages.RDBStorage(
            url=storage_path,
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )
        print("Study created or loaded successfully.")
        # --- Append arguments for the worker command ---
        command_to_run.extend([
            f"--mlflow-tracking-uri={mlflow_tracking_uri}",
        ])

    command_to_run.extend([
        f"--storage={storage_path}",
        f"--study-name={study_name}",
    ])

    print(f"Starting {num_workers} parallel optimization workers for study: '{study_name}'...")
    print(f"Command to be executed by each worker: {' '.join(command_to_run)}")

    # Prepare arguments for the worker pool
    delay = 1  # Delay in seconds between worker starts
    worker_args = [(i, command_to_run, i * delay) for i in range(num_workers)]

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_worker, worker_args)

    print("All workers have completed.")
