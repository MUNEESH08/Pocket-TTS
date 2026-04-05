from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from pocket_tts import TTS
import io
import torchaudio
import torch

app = FastAPI()

# Load once (VERY IMPORTANT for speed)
tts = TTS()

class Request(BaseModel):
    text: str

@app.post("/tts")
async def generate_audio(req: Request):
    text = req.text

    # Generate audio
    audio = tts(text)

    # Handle output (tensor → wav)
    if isinstance(audio, torch.Tensor):
        audio = audio.unsqueeze(0)

    buffer = io.BytesIO()
    torchaudio.save(buffer, audio, 22050, format="wav")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="audio/wav")
