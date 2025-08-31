import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from src.helper.probe_gru_latent_space import flatten_label
from src.utils import DATA_DIR


def load_data_for_tsne(metadata_path: Path, num_samples: int):
    """Loads latent states and labels, and subsamples the data for t-SNE."""
    print(f"Loading data from {metadata_path}...")
    records = []
    with open(metadata_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        raise ValueError("No data found in the metadata file.")

    df = pd.DataFrame(records)

    # Subsample the data before heavy processing
    if len(df) > num_samples:
        print(f"Subsampling data from {len(df)} to {num_samples} points.")
        df = df.sample(n=num_samples, random_state=42)

    latent_states = np.array(df['latent_state'].tolist())
    labels = [flatten_label(r) for r in df['label']]
    df_labels = pd.DataFrame(labels, index=df.index)

    # Reshape the latent states from (n_samples, layers, dim) to (n_samples, layers * dim)
    if latent_states.ndim == 3:
        print(f"Original latent state shape: {latent_states.shape}")
        n_samples_actual = latent_states.shape[0]
        latent_states = latent_states.reshape(n_samples_actual, -1)
        print(f"Reshaped latent state to: {latent_states.shape}")

    print(f"Loaded {len(latent_states)} data points.")
    return latent_states, df_labels


def plot_tsne(
        tsne_results: np.ndarray,
        labels: pd.DataFrame,
        output_dir: Path
):
    """Creates and saves scatter plots for each label."""
    print("\nGenerating plots...")
    output_dir.mkdir(parents=True, exist_ok=True)

    for column in labels.columns:
        print(f"  Plotting for label: {column}")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 10))

        # Check if the column is numeric or categorical for plotting
        if pd.api.types.is_numeric_dtype(labels[column]):
            # Use a continuous color palette
            palette = "viridis"
            legend = "auto"
        else:
            # Use a categorical color palette
            palette = "deep"
            legend = "full"

        sns.scatterplot(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=labels[column],
            palette=palette,
            s=15,  # Smaller points
            alpha=0.7,
            linewidth=0,
            ax=ax,
            legend=legend
        )

        ax.set_title(f"t-SNE of Latent Space, Colored by '{column}'", fontsize=16)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

        # Improve legend for categorical data
        if not pd.api.types.is_numeric_dtype(labels[column]):
            # Shrink current axis and place legend outside
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=column)

        plot_path = output_dir / f"tsne_colored_by_{column}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved plot to {plot_path}")


def main(metadata_path: Path, output_dir: Path, num_samples: int):
    try:
        X, df_labels = load_data_for_tsne(metadata_path, num_samples)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return

    print("\nRunning t-SNE... (This may take a few minutes)")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    tsne_results = tsne.fit_transform(X)
    print("t-SNE computation complete.")

    plot_tsne(tsne_results, df_labels, output_dir)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the latent space of a GRU World Model using t-SNE."
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
        default=str(DATA_DIR / "gru_latent_analysis" / "tsne_visualization"),
        help="Directory to save the output plots."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples to use for t-SNE. Reduces computation time."
    )
    args = parser.parse_args()

    main(Path(args.metadata_path), Path(args.output_dir), args.num_samples)
