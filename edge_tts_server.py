"""
Free Edge-TTS HTTP wrapper for n8n
-----------------------------------
Turns Microsoft Edge's free neural voices into a simple REST endpoint.
No API key, no billing account, no cost - genuinely free forever.

SETUP:
1. pip install fastapi uvicorn edge-tts --break-system-packages
2. python edge_tts_server.py
3. Deploy free on Render.com / Railway.app / any free-tier VM
   (or run locally + use ngrok for a public URL during testing)

USAGE (from n8n HTTP Request node):
POST https://your-deployed-url.com/tts
Body (JSON): { "text": "your script here", "voice": "en-US-GuyNeural" }
Response: raw MP3 audio bytes (set n8n's responseFormat to "file")

VOICE OPTIONS (popular English ones):
  en-US-GuyNeural       - male, US, natural conversational
  en-US-JennyNeural     - female, US, warm/friendly
  en-US-AriaNeural      - female, US, expressive/news-style
  en-GB-RyanNeural      - male, British
  en-GB-SoniaNeural     - female, British
Full list: run `edge-tts --list-voices` after installing
"""

from fastapi import FastAPI, Response
from pydantic import BaseModel
import edge_tts
import asyncio
import io

app = FastAPI()


class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-GuyNeural"   # default voice, change as you like


@app.post("/tts")
async def generate_speech(req: TTSRequest):
    communicate = edge_tts.Communicate(req.text, req.voice)
    audio_buffer = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)
    return Response(content=audio_buffer.read(), media_type="audio/mpeg")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "edge-tts-wrapper"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
