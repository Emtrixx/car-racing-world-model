import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.helper.probe_gru_latent_space import flatten_label
from src.utils import DATA_DIR


def load_data_with_images(metadata_path: Path, image_base_dir: Path):
    """Loads metadata and prepares image paths."""
    print(f"Loading data from {metadata_path}...")
    records = []
    with open(metadata_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        raise ValueError("No data found in the metadata file.")

    df = pd.DataFrame(records)
    labels_df = pd.DataFrame([flatten_label(r['label']) for r in records])
    df = pd.concat([df.drop(columns=['label', 'latent_state']), labels_df], axis=1)

    # Create absolute image paths
    df['abs_frame_path'] = df['frame_path'].apply(lambda p: image_base_dir / p)

    print(f"Loaded {len(df)} records.")
    return df


def get_average_image(image_paths: pd.Series):
    """Computes the average image from a list of paths."""
    if image_paths.empty:
        return None

    sum_image = None
    count = 0

    for img_path in tqdm(image_paths, desc="Averaging images", leave=False):
        img = cv2.imread(str(img_path))
        if img is not None:
            # Convert to float64 for safe summation
            if sum_image is None:
                sum_image = np.zeros_like(img, dtype=np.float64)
            sum_image += img.astype(np.float64)
            count += 1

    if count == 0:
        return None

    avg_image = (sum_image / count).astype(np.uint8)
    return cv2.cvtColor(avg_image, cv2.COLOR_BGR2RGB)


def visualize_label_consistency(
        df: pd.DataFrame,
        label_col: str,
        output_dir: Path
):
    """Generates a side-by-side comparison plot for a given label."""
    print(f"--- Visualizing consistency for: {label_col} ---")

    groups = {}
    # Group by the unique categorical values in the column
    grouped = df.groupby(label_col)

    for name, group in grouped:
        print(f"  Processing group: '{name}' ({len(group)} images)")
        avg_img = get_average_image(group['abs_frame_path'])
        if avg_img is not None:
            groups[name] = avg_img

    if not groups:
        print("No images found or processed. Skipping plot.")
        return

    # Sort groups by name for consistent plot order (optional but good practice)
    sorted_groups = sorted(groups.items(), key=lambda item: str(item[0]))

    n_groups = len(sorted_groups)
    if n_groups == 0:
        return

    # Create and save plot
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6), squeeze=False)
    fig.suptitle(f"Label Consistency Check for '{label_col}'", fontsize=20)

    for i, (name, avg_img) in enumerate(sorted_groups):
        ax = axes[0, i]
        ax.imshow(avg_img)
        num_images_in_group = grouped.get_group(name).shape[0]
        ax.set_title(f"Group: {name}\n({num_images_in_group} images)", fontsize=14)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = output_dir / f"consistency_{label_col}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved comparison plot to {plot_path}")


def main(metadata_path: Path, image_base_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to {output_dir}")

    try:
        df = load_data_with_images(metadata_path, image_base_dir)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return

    # --- Define which labels to check ---
    # All new labels are discrete categories
    labels_to_check = [
        'car_position',
        'car_angle',
        'car_is_off_track',
        'track_curvature_current',
        'track_curvature_upcoming',
        'track_distance_to_turn'
    ]

    for label in labels_to_check:
        if label in df.columns:
            visualize_label_consistency(df.copy(), label, output_dir)
        else:
            print(f"Warning: Label column '{label}' not found in data. Skipping.")

    print("\nVisualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visually check the consistency of VLM-generated labels by averaging images."
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(DATA_DIR / "gru_latent_analysis" / "metadata.jsonl"),
        help="Path to the metadata.jsonl file from the analysis script."
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default=str(DATA_DIR / "gru_latent_analysis"),
        help="The base directory where the rollout images are stored."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR / "gru_latent_analysis" / "label_consistency_checks"),
        help="Directory to save the output plots."
    )
    args = parser.parse_args()

    main(Path(args.metadata_path), Path(args.image_base_dir), Path(args.output_dir))
