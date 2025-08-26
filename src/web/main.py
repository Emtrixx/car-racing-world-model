
import base64
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel

from src.web.in_memory_dream import Dreamer

dapp = FastAPI()

dapp.mount("/static", StaticFiles(directory="src/web/static"), name="static")

dreamer = None

class Action(BaseModel):
    action: list[float]

@dapp.get("/")
async def read_index():
    return FileResponse('src/web/static/index.html')

@dapp.post("/api/v1/dream/start/{model_type}")
async def start_dream(model_type: str):
    global dreamer
    try:
        dreamer = Dreamer(model_type=model_type)
        initial_frame = dreamer.start()
        return {"frame": base64.b64encode(initial_frame).decode('utf-8')}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@dapp.post("/api/v1/dream/step")
async def step_dream(action: Action):
    global dreamer
    if dreamer is None:
        raise HTTPException(status_code=400, detail="Dream not started")
    
    try:
        next_frame = dreamer.step(action.action)
        return {"frame": base64.b64encode(next_frame).decode('utf-8')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
