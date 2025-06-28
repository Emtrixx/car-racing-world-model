import json
import os
from pathlib import Path

import torch
from PIL import Image
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

from utils import VQ_VAE_CHECKPOINT_FILENAME, DEVICE
from vq_conv_vae import VQVAE, EMBEDDING_DIM, NUM_EMBEDDINGS

# --- Configuration ---

# Define the output directory relative to where you run this script.
# This should point to the `public` folder of your Vite project.
OUTPUT_DIR = Path("../assets")
PATCHES_DIR = OUTPUT_DIR / "decoded_patches"


def save_tensor_as_image(tensor, filename):
    """
    Saves a PyTorch tensor as a grayscale image file.
    Args:
        tensor (torch.Tensor): The tensor to save. Shape [1, C, H, W]
        filename (str or Path): The path to save the image to.
    """
    # Make sure tensor is on CPU and remove batch dimension
    image_tensor = tensor.cpu().squeeze(0)
    # Convert to a grid if there are multiple channels (for visualization)
    # For grayscale, it will just be the single channel.
    grid = make_grid(image_tensor, normalize=True)
    # Convert to NumPy array
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save(filename)


# --- Main Execution ---
if __name__ == '__main__':
    # --- Setup Directories ---
    print(f"Creating output directories at: {OUTPUT_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PATCHES_DIR.mkdir(exist_ok=True)

    # --- Load Model and Codebook ---
    model = VQVAE().to(DEVICE)

    # Check if a trained model file exists, otherwise use random weights
    if os.path.exists(VQ_VAE_CHECKPOINT_FILENAME):
        print(f"Loading trained model from: {VQ_VAE_CHECKPOINT_FILENAME}")
        model.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=DEVICE))
    else:
        print("WARNING: Model file not found. Using randomly initialized weights.")
        print("Please train your model and save it as 'vq_vae_model.pth'.")

    model.eval()
    codebook_weights = model.vq_layer.embedding.weight.data.cpu().numpy()

    # --- Project Codebook to 2D using t-SNE ---
    print("Projecting codebook to 2D using t-SNE... (This may take a moment)")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    projected_vectors = tsne.fit_transform(codebook_weights)
    print("Projection complete.")

    # --- Decode Each Vector and Save Image ---
    print(f"Decoding {NUM_EMBEDDINGS} codebook vectors to image patches...")
    codebook_data = []

    # The feature map size of the encoder is 4x4 (64 -> 32 -> 16 -> 8 -> 4)
    # So the decoder expects an input of shape [1, embedding_dim, 4, 4]
    # We will decode each vector as if it were a full 4x4 feature map of that single vector.

    for i in range(len(codebook_weights)):
        # Get the vector and move it to the correct device
        vector = torch.tensor(codebook_weights[i], device=DEVICE).view(1, EMBEDDING_DIM, 1, 1)

        # Expand the vector to match the expected feature map size (e.g., 4x4)
        # The decoder was trained on feature maps of a certain spatial size.
        # We need to replicate the single vector across that spatial dimension.
        # The encoder output is 4x4, so we expand to that.
        decoder_input = vector.repeat(1, 1, 4, 4)

        with torch.no_grad():
            decoded_patch = model.decoder(decoder_input)

        # Define the path for the frontend (relative path with forward slashes)
        # assets folder is symlinked to the public folder and renamed to "data" in the frontend.
        image_relative_path = f"/data/decoded_patches/patch_{i}.png"

        # Define the full path for saving the file
        image_save_path = OUTPUT_DIR.parent / image_relative_path.lstrip('/')

        save_tensor_as_image(decoded_patch, image_save_path)

        # --- Assemble JSON data ---
        codebook_data.append({
            "index": i,
            "x": float(projected_vectors[i, 0]),
            "y": float(projected_vectors[i, 1]),
            "imagePath": image_relative_path
        })

        print(f"  - Decoded and saved patch {i + 1}/{NUM_EMBEDDINGS}", end='\r')

    print("\nDecoding complete.")

    # --- Save the JSON File ---
    json_path = OUTPUT_DIR / "codebook_data.json"
    print(f"Saving data to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(codebook_data, f, indent=2)

    print("\nAsset generation complete!")
    print(f"The assets are in {OUTPUT_DIR.resolve()}")
