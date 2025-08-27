import os
import torch
import numpy as np
import cv2
from collections import deque
from pathlib import Path

from src.utils import (
    WM_CHECKPOINT_FILENAME_GRU, VQ_VAE_CHECKPOINT_FILENAME, ACTION_DIM, DATA_DIR,
    WM_CHECKPOINT_FILENAME_TRANSFORMER, preprocess_observation
)
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM, VQVAE_NUM_EMBEDDINGS
from src.world_model import WorldModelGRU
from src.transformer_world_model import WorldModelTransformer
from src.dreaming_render import get_starting_state_from_sequence


class Dreamer:
    def __init__(self, model_type='gru', device=None):
        self.model_type = model_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vq_vae = None
        self.world_model = None
        self.hidden_state = None
        self.current_tokens = None
        self.action_history = None
        self.token_history = None

        self._load_models()

    def _load_models(self):
        print(f"Loading models for {self.model_type}...")
        self.vq_vae = VQVAE(embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS).to(self.device)
        self.vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=self.device))
        self.vq_vae.eval()

        if self.model_type == 'gru':
            self.world_model = WorldModelGRU(
                latent_dim=VQVAE_EMBEDDING_DIM,
                action_dim=ACTION_DIM,
            ).to(self.device)
            self.world_model = torch.compile(self.world_model)
            self.world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=self.device))
        elif self.model_type == 'transformer':
            self.world_model = WorldModelTransformer(
                vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
                action_dim=ACTION_DIM,
            ).to(self.device)
            self.world_model = torch.compile(self.world_model)
            self.world_model.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_TRANSFORMER, map_location=self.device))
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        self.world_model.eval()

    def start(self):
        if self.model_type == 'gru':
            return self._start_gru()
        else:
            return self._start_transformer()

    def _start_gru(self):
        INIT_FRAMES_DIR = DATA_DIR / "init_frames"
        if not INIT_FRAMES_DIR.exists():
            raise FileNotFoundError(f"Initial frames directory not found at {INIT_FRAMES_DIR}")

        image_files = sorted([os.path.join(INIT_FRAMES_DIR, f) for f in os.listdir(INIT_FRAMES_DIR)])
        if not image_files:
            raise FileNotFoundError(f"No images found in {INIT_FRAMES_DIR}")

        priming_sequence = image_files[:10]
        self.hidden_state, initial_frame_tensor, self.current_tokens = get_starting_state_from_sequence(
            priming_sequence, self.world_model, self.vq_vae, self.device
        )
        self.current_tokens = self.current_tokens.to(self.device)
        return self._tensor_to_image(initial_frame_tensor)

    def _start_transformer(self):
        init_image_path = DATA_DIR / "init_frames/frame_0001.png"
        if not os.path.exists(init_image_path):
            raise FileNotFoundError(f"Initial image not found at {init_image_path}")

        frame = cv2.imread(str(init_image_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = preprocess_observation(frame_rgb)
        frame_tensor = torch.tensor(processed_frame, dtype=torch.float32, device=self.device)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            z_continuous = self.vq_vae.encoder(frame_tensor)
            z_continuous = self.vq_vae._pre_vq_conv(z_continuous)
            _, _, _, initial_tokens = self.vq_vae.vq_layer(z_continuous)

        initial_tokens = initial_tokens.view(-1)

        self.action_history = deque(maxlen=32)
        self.token_history = deque(maxlen=32)
        zero_action = torch.zeros(ACTION_DIM, device=self.device)

        for _ in range(32):
            self.action_history.append(zero_action)
            self.token_history.append(initial_tokens)

        return self._tensor_to_image(frame_tensor)

    def step(self, action):
        if self.model_type == 'gru':
            return self._step_gru(action)
        else:
            return self._step_transformer(action)

    def _step_gru(self, action):
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred_logits, _, _, next_hidden_state, _ = self.world_model(
                obs_tokens=self.current_tokens,
                actions=action_tensor,
                initial_hidden_state=self.hidden_state
            )
            pred_logits = pred_logits.squeeze(1)
            predicted_indices = torch.distributions.Categorical(logits=pred_logits).sample()
            self.current_tokens = predicted_indices.unsqueeze(1)

            b, h, w = 1, self.world_model.grid_size, self.world_model.grid_size
            quantized_vectors = self.vq_vae.vq_layer.embeddings.data[predicted_indices]
            quantized_grid = quantized_vectors.view(b, h, w, -1)
            quantized_grid_permuted = quantized_grid.permute(0, 3, 1, 2)
            decoded_image = self.vq_vae.decoder(quantized_grid_permuted)

            self.hidden_state = next_hidden_state
            return self._tensor_to_image(decoded_image)

    def _step_transformer(self, action):
        current_action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.action_history.append(current_action_tensor)

        with torch.no_grad():
            action_history_tensor = torch.stack(list(self.action_history)).unsqueeze(0)
            token_history_tensor = torch.stack(list(self.token_history)).unsqueeze(0)

            pred_logits, _, _, generated_tokens = self.world_model(
                action_history_tensor, token_history_tensor
            )

            b, h, w, c = pred_logits.shape
            pred_probs = torch.softmax(pred_logits.view(-1, c), dim=-1)
            tokens_for_decoding = torch.multinomial(pred_probs, num_samples=1).squeeze(1)

            quantized_vectors = self.vq_vae.vq_layer.embeddings[tokens_for_decoding]
            quantized_grid = quantized_vectors.view(h, w, -1)
            quantized_grid_permuted = quantized_grid.permute(2, 0, 1).unsqueeze(0)
            decoded_image = self.vq_vae.decoder(quantized_grid_permuted)

            self.token_history.append(generated_tokens.squeeze(0))
            return self._tensor_to_image(decoded_image)

    def _tensor_to_image(self, tensor):
        image_np = (tensor.squeeze(0).permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        if image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', image_np)
        return buffer.tobytes()
