import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from src.utils import DATA_DIR


def flatten_label(label_dict):
    """Flattens the nested label dictionary."""
    flat = {}
    if "car_state" in label_dict:
        for k, v in label_dict["car_state"].items():
            flat[f"car_{k}"] = v
    if "track_state" in label_dict:
        for k, v in label_dict["track_state"].items():
            flat[f"track_{k}"] = v
    return flat


def load_data(metadata_path: Path):
    """Loads latent states and labels from the metadata file."""
    print(f"Loading data from {metadata_path}...")
    records = []
    with open(metadata_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        raise ValueError("No data found in the metadata file.")

    latent_states = np.array([r['latent_state'] for r in records])

    # Reshape the latent states from (n_samples, layers, dim) to (n_samples, layers * dim)
    if latent_states.ndim == 3:
        print(f"Original latent state shape: {latent_states.shape}")
        n_samples = latent_states.shape[0]
        latent_states = latent_states.reshape(n_samples, -1)
        print(f"Reshaped latent state to: {latent_states.shape}")

    labels = [flatten_label(r['label']) for r in records]

    df_labels = pd.DataFrame(labels)

    print(f"Loaded {len(latent_states)} data points.")
    print("Available labels to analyze:", df_labels.columns.tolist())
    return latent_states, df_labels


def run_probe(
        X: np.ndarray,
        y: pd.Series,
        target_name: str,
        output_dir: Path,
        model_type: str
):
    """
    Trains a regression model for a single target, evaluates it,
    and saves a visualization.
    """
    probe_type = "MLP" if model_type == 'mlp' else "Linear"
    print(f"--- {probe_type} Probing for: {target_name} ---")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Select and train model
    if model_type == 'linear':
        model = LinearRegression()
    else:  # mlp
        # Using a 2-layer MLP. For better results, this could be tuned.
        model = MLPRegressor(hidden_layer_sizes=(128, 128), random_state=42, max_iter=500, early_stopping=True)
        print("  (Using MLPRegressor with 2 hidden layers of size 128)")

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")

    # Create and save plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6, edgecolor='k')
    # Add the y=x line for reference
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Perfect Prediction")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{probe_type} Probe: '{target_name}'\n$R^2 = {r2:.3f}$ | MSE = {mse:.3f}")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plot_path = output_dir / f"{model_type}_probe_{target_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot to {plot_path}")


def main(metadata_path: Path, output_dir: Path, model_type: str):
    """
    Main function to run the linear probing analysis.
    """
    # Adjust output dir based on model type
    output_dir = output_dir / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to {output_dir}")
    print(f"Using model type: {model_type.upper()}")

    try:
        X, df_labels = load_data(metadata_path)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return

    for column in df_labels.columns:
        y = df_labels[column]
        run_probe(X, y, column, output_dir, model_type)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform linear or non-linear (MLP) probing on a GRU World Model's latent space."
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(DATA_DIR / "gru_latent_analysis" / "metadata.jsonl"),
        help="Path to the metadata.jsonl file from the analysis script."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR / "gru_latent_analysis" / "probe_results"),
        help="Directory to save the output plots."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=['linear', 'mlp'],
        default='linear',
        help="Type of model to use for probing: 'linear' for Linear Regression, 'mlp' for a non-linear MLP."
    )
    args = parser.parse_args()

    main(Path(args.metadata_path), Path(args.output_dir), args.model_type)
