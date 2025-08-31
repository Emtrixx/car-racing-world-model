import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import IMAGES_DIR


def visualize_positional_embedding(
        model_path: str,
        output_path: str,
        grid_size: int
):
    """
    Loads a model state dictionary, extracts the grid positional embedding,
    and visualizes its pairwise cosine similarity as a heatmap.
    """
    print(f"Loading model state_dict from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model file: {e}")
        return

    # --- Extract the Embedding --- 
    embedding_key = '_orig_mod.grid_pos_embedding'  # key for compiled models
    if embedding_key not in state_dict:
        print(f"Error: Could not find '{embedding_key}' in the model's state_dict.")
        print(f"Available keys: {list(state_dict.keys())}")
        return

    pos_embedding = state_dict[embedding_key].squeeze(0)  # Shape: [num_tokens, embed_dim]
    num_tokens, embed_dim = pos_embedding.shape

    print(f"Successfully loaded '{embedding_key}' with shape: {pos_embedding.shape}")

    if num_tokens != grid_size * grid_size:
        print(f"Warning: Number of tokens in embedding ({num_tokens}) does not match "
              f"the expected number from grid_size ({grid_size * grid_size}).")

    # --- Calculate Cosine Similarity --- 
    print("Calculating pairwise cosine similarity...")
    # The tensor might be on GPU, move to cpu and convert to numpy
    similarity_matrix = cosine_similarity(pos_embedding.cpu().numpy())

    # --- Plot the Heatmap --- 
    print(f"Generating heatmap...")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='viridis')

    # --- Formatting --- 
    ax.set_title(f"Cosine Similarity of {grid_size}x{grid_size} Grid Positional Embedding", fontsize=16)
    ax.set_xlabel("Token Index", fontsize=12)
    ax.set_ylabel("Token Index", fontsize=12)

    # Add a color bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", fontsize=12)

    # Set ticks to be at every grid line
    ax.set_xticks(np.arange(0, num_tokens, grid_size))
    ax.set_yticks(np.arange(0, num_tokens, grid_size))
    ax.grid(which='major', color='w', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # --- Save the Figure --- 
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)

    print(f"Visualization saved successfully to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the grid positional embedding from a trained world model."
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the trained model checkpoint (.pt or .pth file containing the state_dict)."
    )
    parser.add_argument(
        "--output-path", type=str, default=IMAGES_DIR / "positional_embedding_similarity.png",
        help="Path to save the output heatmap image."
    )
    parser.add_argument(
        "--grid-size", type=int, default=4,
        help="The side length of the token grid."
    )
    args = parser.parse_args()

    visualize_positional_embedding(
        model_path=args.model_path,
        output_path=args.output_path,
        grid_size=args.grid_size
    )
