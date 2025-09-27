import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


def load_evaluation_data(log_patterns):
    """
    Loads evaluation data from npz files matching the given patterns.

    Args:
        log_patterns (dict): A dictionary mapping algorithm names to glob patterns
                             for their evaluation files.

    Returns:
        A dictionary mapping algorithm names to lists of (timesteps, mean_rewards) tuples.
    """
    all_results = {}
    for algorithm, pattern in log_patterns.items():
        all_results[algorithm] = []
        file_paths = glob.glob(pattern)
        if not file_paths:
            print(f"Warning: No files found for algorithm '{algorithm}' with pattern '{pattern}'")
            continue

        print(f"Found {len(file_paths)} files for {algorithm}...")
        for file_path in file_paths:
            try:
                data = np.load(file_path)
                timesteps = data['timesteps']
                # Calculate mean reward for each evaluation step
                mean_rewards = np.mean(data['results'], axis=1)
                all_results[algorithm].append((timesteps, mean_rewards))
            except Exception as e:
                print(f"Error loading or processing {file_path}: {e}")
    return all_results


def align_data(all_results, max_timesteps, num_points=100):
    """
    Aligns evaluation data to a common set of timesteps using interpolation.

    Args:
        all_results (dict): Data loaded by `load_evaluation_data`.
        max_timesteps (int): The maximum timestep for the x-axis.
        num_points (int): The number of points to interpolate to.

    Returns:
        A dictionary mapping algorithm names to a NumPy array of shape
        (num_runs, num_points) containing the interpolated scores.
    """
    aligned_scores = {}
    common_timesteps = np.linspace(0, max_timesteps, num_points)

    for algorithm, run_data in all_results.items():
        if not run_data:
            continue

        interpolated_runs = []
        for timesteps, mean_rewards in run_data:
            # Ensure that the first timestep is 0 for correct interpolation
            if timesteps[0] != 0:
                timesteps = np.insert(timesteps, 0, 0)
                # Use the first available reward as the reward at timestep 0
                mean_rewards = np.insert(mean_rewards, 0, mean_rewards[0])

            # Interpolate
            interpolated_rewards = np.interp(common_timesteps, timesteps, mean_rewards)
            interpolated_runs.append(interpolated_rewards)

        aligned_scores[algorithm] = np.array(interpolated_runs)

    return aligned_scores, common_timesteps


def main():
    """
    Main function to run the analysis and generate plots.
    """
    # --- Configuration ---
    LOG_DIR = Path("logs/sb3_logs")
    RESULTS_DIR = Path("results/rliable_analysis")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    MAX_TIMESTEPS = 2_000_000  # Adjust as per your longest training runs

    # Define glob patterns for the evaluation files of each algorithm
    # Assumes a directory structure like: `logs/sb3_logs/{run_name}_eval/evaluations.npz`
    # The `run_name` should be consistent across runs of the same algorithm type.
    log_patterns = {
        "PPO_SB3": str(LOG_DIR / "cnn_sb3_ppo_*_eval/evaluations.npz"),
        "Dream_GRU": str(LOG_DIR / "ppo_dream_gru_*_eval/evaluations.npz"),
        "Dream_Transformer": str(LOG_DIR / "ppo_dream_transformer_*_eval/evaluations.npz"),
        "Dyna_GRU": str(LOG_DIR / "dyn_gru_*_eval/evaluations.npz"),
        "Dyna_Transformer": str(LOG_DIR / "dyn_transformer_*_eval/evaluations.npz"),
        # "PPO_SB3": str(LOG_DIR / "*/evaluations.npz"),
        # "Dream_GRU": str(LOG_DIR / "*/evaluations.npz"),
        # "Dream_Transformer": str(LOG_DIR / "*/evaluations.npz"),
        # "Dyna_GRU": str(LOG_DIR / "*/evaluations.npz"),
        # "Dyna_Transformer": str(LOG_DIR / "*/evaluations.npz"),
    }

    # --- Load and Process Data ---
    raw_results = load_evaluation_data(log_patterns)

    # Check if any data was loaded
    if not any(raw_results.values()):
        print("No evaluation data was found. Please run the experiments first.")
        print("Expected file paths like: logs/sb3_logs/ppo_sb3_seed1_eval/evaluations.npz")
        return

    scores, common_timesteps = align_data(raw_results, MAX_TIMESTEPS)

    if not scores:
        print("Data alignment failed. No scores to analyze.")
        return

    # --- Analyze and Plot ---
    print("--- Generating RLiable Plots and Metrics ---")

    # Set plot style
    sns.set_style("whitegrid")

    # 1. Aggregate Learning Curves (Sample Efficiency Curves) - COMPARISON PLOT
    algorithms_to_plot = list(scores.keys())
    # Calculate IQM and CIs for plot (for all algorithms together)
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x[..., i]) for i in range(x.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores, iqm_fn, reps=50000)

    ax = plot_utils.plot_sample_efficiency_curve(
        common_timesteps,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms_to_plot,
        xlabel="Timesteps",
        ylabel="IQM Episode Reward",
        figsize=(10, 6),
    )
    plt.title("Sample Efficiency Curves (All Algorithms)")
    plt.tight_layout()
    save_path = RESULTS_DIR / "sample_efficiency_curves_all.png"
    fig = ax.get_figure()
    plt.savefig(save_path, dpi=300)
    print(f"Saved combined sample efficiency curves to {save_path}")
    plt.close(fig)

    # 1b. INDIVIDUAL ALGORITHM PLOTS
    print("Generating individual algorithm plots...")
    for alg in algorithms_to_plot:
        # Skip if somehow missing
        if alg not in iqm_scores:
            continue
        single_iqm_scores = {alg: iqm_scores[alg]}
        single_iqm_cis = {alg: iqm_cis[alg]}
        ax_single = plot_utils.plot_sample_efficiency_curve(
            common_timesteps,
            single_iqm_scores,
            single_iqm_cis,
            algorithms=[alg],
            xlabel="Timesteps",
            ylabel="IQM Episode Reward",
            figsize=(8, 5),
        )
        plt.title(f"Sample Efficiency Curve - {alg}")
        plt.tight_layout()
        indiv_path = RESULTS_DIR / f"sample_efficiency_curve_{alg}.png"
        fig_single = ax_single.get_figure()
        plt.savefig(indiv_path, dpi=300)
        print(f"  Saved {alg} IQM curve to {indiv_path}")
        plt.close(fig_single)

        # Additional plot: all runs + mean + IQM for more detailed per-algorithm inspection
        runs_array = scores[alg]  # shape (num_runs, num_points)
        if runs_array.shape[0] > 0:
            fig_runs, ax_runs = plt.subplots(figsize=(8, 5))
            for i, run_curve in enumerate(runs_array):
                ax_runs.plot(common_timesteps, run_curve, color='gray', alpha=0.25)
            mean_curve = runs_array.mean(axis=0)
            ax_runs.plot(common_timesteps, mean_curve, label='Mean', color='blue', linewidth=2)
            iqm_curve = single_iqm_scores[alg]
            ax_runs.plot(common_timesteps, iqm_curve, label='IQM', color='orange', linewidth=2, linestyle='--')
            ax_runs.set_title(f"Runs, Mean & IQM - {alg}")
            ax_runs.set_xlabel("Timesteps")
            ax_runs.set_ylabel("Episode Reward")
            ax_runs.legend()
            plt.tight_layout()
            runs_path = RESULTS_DIR / f"runs_mean_iqm_{alg}.png"
            plt.savefig(runs_path, dpi=300)
            print(f"  Saved {alg} runs + mean + IQM plot to {runs_path}")
            plt.close(fig_runs)

    # 2. Calculate and Print Aggregate Metrics
    # This calculates a single IQM score over all runs and timesteps for a final summary
    print("\n--- Aggregate Metrics (IQM) ---")
    # Pass the scores directly (2D array) to get a single aggregate value and CIs
    aggregate_scores, aggregate_cis = rly.get_interval_estimates(scores, metrics.aggregate_iqm)

    # Create a formatted table
    header = f"| {'Algorithm':<20} | {'IQM':<10} | {'95% CI Lower':<15} | {'95% CI Upper':<15} |"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for algorithm, mean_score in aggregate_scores.items():
        lower_ci, upper_ci = aggregate_cis[algorithm]
        print(f"| {algorithm:<20} | {mean_score.item():<10.2f} | {lower_ci.item():<15.2f} | {upper_ci.item():<15.2f} |")

    print(separator)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
