import base64
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from src.transformer_world_model import WorldModelTransformer
from src.utils import (
    WM_CHECKPOINT_FILENAME_GRU, VQ_VAE_CHECKPOINT_FILENAME, ACTION_DIM,
    WM_CHECKPOINT_FILENAME_TRANSFORMER
)
from src.vq_conv_vae import VQVAE, VQVAE_EMBEDDING_DIM, VQVAE_NUM_EMBEDDINGS
from src.web.in_memory_dream import Dreamer
from src.world_model import WorldModelGRU

models = {}
sessions = {}
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading VQ-VAE model...")
    vq_vae = VQVAE(embedding_dim=VQVAE_EMBEDDING_DIM, num_embeddings=VQVAE_NUM_EMBEDDINGS).to(device)
    vq_vae.load_state_dict(torch.load(VQ_VAE_CHECKPOINT_FILENAME, map_location=device))
    vq_vae.eval()
    models['vq_vae'] = vq_vae
    print("VQ-VAE model loaded.")

    print("Loading GRU world model...")
    gru_wm = WorldModelGRU(
        latent_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
    ).to(device)
    gru_wm = torch.compile(gru_wm)
    gru_wm.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_GRU, map_location=device))
    gru_wm.eval()
    models['gru'] = gru_wm
    print("GRU world model loaded.")

    print("Loading Transformer world model...")
    transformer_wm = WorldModelTransformer(
        vqvae_embed_dim=VQVAE_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
    ).to(device)
    transformer_wm = torch.compile(transformer_wm)
    transformer_wm.load_state_dict(torch.load(WM_CHECKPOINT_FILENAME_TRANSFORMER, map_location=device))
    transformer_wm.eval()
    models['transformer'] = transformer_wm
    print("Transformer world model loaded.")

    yield

    # Clean up models and sessions
    models.clear()
    sessions.clear()


dapp = FastAPI(lifespan=lifespan)

dapp.mount("/static", StaticFiles(directory="src/web/static"), name="static")


@dapp.get("/")
async def read_index():
    return FileResponse('src/web/static/index.html')


@dapp.post("/api/v1/dream/start/{model_type}")
async def start_dream_session(model_type: str):
    if model_type not in models:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

    session_id = str(uuid.uuid4())
    try:
        dreamer = Dreamer(
            model_type=model_type,
            world_model=models[model_type],
            vq_vae=models['vq_vae'],
            device=device
        )
        sessions[session_id] = dreamer
        return {"session_id": session_id}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=500, detail=str(e))


@dapp.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if session_id not in sessions:
        await websocket.close(code=4000, reason="Session not found")
        return

    dreamer = sessions[session_id]
    await websocket.accept()

    try:
        # Send initial frame
        initial_frame = dreamer.start()
        await websocket.send_json({"frame": base64.b64encode(initial_frame).decode('utf-8')})

        while True:
            data = await websocket.receive_json()
            if data['type'] == 'step':
                action = data['action']
                next_frame = dreamer.step(action)
                await websocket.send_json({"frame": base64.b64encode(next_frame).decode('utf-8')})
            elif data['type'] == 'reset':
                initial_frame = dreamer.start()
                await websocket.send_json({"frame": base64.b64encode(initial_frame).decode('utf-8')})

    except WebSocketDisconnect:
        print(f"Client disconnected from session {session_id}")
    except Exception as e:
        print(f"Error in session {session_id}: {e}")
    finally:
        # Clean up session
        if session_id in sessions:
            del sessions[session_id]
            print(f"Cleaned up session {session_id}")
