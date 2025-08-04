import subprocess
import multiprocessing
import argparse

# =============================================================================
#  CONFIGURATION
# =============================================================================
CONFIGS = {
    "ppo": {
        "description": "Hyperparameter sweep for PPO on an A100 GPU.",
        "num_workers": 16,
        "command_args": [
            "python",
            "-m",
            "src.train_ppo_sb3",
            "--optimize",
        ]
    },
}


# =============================================================================

def run_worker(worker_id_and_command):
    """Function to be executed by each worker process."""
    worker_id, command = worker_id_and_command
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
        formatter_class=argparse.RawTextHelpFormatter  # For better help text formatting
    )
    parser.add_argument(
        "config",
        type=str,
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

    args = parser.parse_args()

    # --- Select the configuration ---
    selected_config = CONFIGS[args.config]
    command_to_run = selected_config["command_args"]
    num_workers = args.num_workers if args.num_workers is not None else selected_config["num_workers"]

    print(f"Starting {num_workers} parallel optimization workers for config: '{args.config}'...")
    print(f"Command to be executed by each worker: {' '.join(command_to_run)}")

    # Prepare arguments for the worker pool
    worker_args = [(i, command_to_run) for i in range(num_workers)]

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_worker, worker_args)

    print("All workers have completed.")
