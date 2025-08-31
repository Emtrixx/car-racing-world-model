import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

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


def run_classification_probe(
        X: np.ndarray,
        y: pd.Series,
        target_name: str,
        output_dir: Path,
        model_type: str
):
    """Trains a classification model, evaluates it, and saves a confusion matrix."""
    probe_type = "MLP" if model_type == 'mlp' else "Linear (Logistic)"
    print(f"--- {probe_type} Probing for: {target_name} ---")

    # Drop rows where target is None/NaN
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]

    if len(y.unique()) < 2:
        print(f"  Skipping {target_name} because it has fewer than 2 unique classes.")
        return

    # Encode string labels to integers for model training
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Select and train model
    if model_type == 'linear':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:  # mlp
        model = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42, max_iter=500, early_stopping=True)

    print(f"  Training {model.__class__.__name__}...")
    model.fit(X_train, y_train_encoded)

    # Evaluate model
    y_pred_encoded = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

    # Use original string labels for the classification report for readability
    y_test_labels = le.inverse_transform(y_test_encoded)
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    report = classification_report(y_test_labels, y_pred_labels, labels=le.classes_)

    print(f"  Accuracy: {accuracy:.4f}")
    print("  Classification Report:")
    print("    " + report.replace('\n', '\n    '))

    # Create and save plot (Confusion Matrix)
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test_encoded, ax=ax, xticks_rotation='vertical',
            cmap='Blues', display_labels=le.classes_
        )
        ax.set_title(f"{probe_type} Probe: '{target_name}'\nAccuracy = {accuracy:.3f}")

        plot_path = output_dir / f"{model_type}_probe_confusion_matrix_{target_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved confusion matrix to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot for {target_name}: {e}")


def main(metadata_path: Path, output_dir: Path, model_type: str):
    """
    Main function to run the probing analysis.
    """
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
        run_classification_probe(X, y, column, output_dir, model_type)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform linear or non-linear classification probing on a GRU World Model's latent space."
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
        help="Type of model to use for probing: 'linear' for Logistic Regression, 'mlp' for MLPClassifier."
    )
    args = parser.parse_args()

    main(Path(args.metadata_path), Path(args.output_dir), args.model_type)
