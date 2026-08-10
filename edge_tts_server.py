"""
Free Edge-TTS + Render HTTP wrapper for n8n
--------------------------------------------
Provides two endpoints:
  POST /tts    – generate MP3 voiceover via Microsoft Edge TTS (free, no key)
  POST /render – stitch voiceover + background video into a 1080×1920 Short
                 using FFmpeg, returns the finished MP4.

SETUP:
  Docker (recommended for /render – FFmpeg required):
    docker build -t shorts-auto . && docker run -p 8000:8000 shorts-auto

  Local (TTS-only, no FFmpeg):
    pip install fastapi uvicorn edge-tts python-multipart
    python edge_tts_server.py

USAGE:
  POST /tts
    Body (JSON): { "text": "...", "voice": "en-US-GuyNeural" }
    Returns: MP3 audio bytes

  POST /render  (multipart/form-data)
    Fields: audio (file), video (file), title (text), caption (text)
    Returns: MP4 video

VOICE OPTIONS:
  en-US-GuyNeural   – male, US, conversational
  en-US-JennyNeural – female, US, warm
  en-US-AriaNeural  – female, US, expressive
  en-GB-RyanNeural  – male, British
  en-GB-SoniaNeural – female, British
  Full list: edge-tts --list-voices
"""

import io
import os
import shutil
import subprocess
import tempfile
import textwrap

import edge_tts
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

app = FastAPI(title="Shorts Auto", version="2.0.0")

# Path to DejaVu Bold font – present when the Docker image installs fonts-dejavu-core
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _esc(text: str) -> str:
    """Escape characters that break ffmpeg drawtext filter values."""
    return (
        text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\u2019")   # curly apostrophe – safe in drawtext
            .replace("%", "\\%")
    )


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


@app.post("/render")
async def render(
    audio: UploadFile = File(..., description="Voiceover MP3 from /tts"),
    video: UploadFile = File(..., description="Background footage clip"),
    title: str = Form("", description="Short title (used if caption is empty)"),
    caption: str = Form("", description="Caption text to overlay on the video"),
):
    """
    Stitch voiceover audio + background video into a 1080×1920 YouTube Short.

    The background is looped/cropped to fill the vertical frame, the caption
    is drawn at the bottom with a semi-transparent box, and the output is
    trimmed to the length of the voiceover audio.
    """
    tmp = tempfile.mkdtemp()
    bg_path = os.path.join(tmp, "bg.mp4")
    voice_path = os.path.join(tmp, "voice.mp3")
    out_path = os.path.join(tmp, "out.mp4")
    cap_path = os.path.join(tmp, "caption.txt")

    # Write uploaded files to temp directory
    with open(bg_path, "wb") as f:
        f.write(await video.read())
    with open(voice_path, "wb") as f:
        f.write(await audio.read())

    # Wrap caption text to ~28 chars per line so it fits 1080px width at 54pt
    display_text = caption.strip() or title.strip() or "Watch till the end!"
    wrapped = "\n".join(textwrap.wrap(_esc(display_text), width=28))
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(wrapped)

    # Build FFmpeg video-filter chain:
    #   1. Scale so shortest side ≥ 1080×1920, then centre-crop to exact size
    #   2. Draw caption text near the bottom with a translucent backing box
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        f"drawtext=fontfile={FONT}:textfile={cap_path}:reload=0:"
        "fontcolor=white:fontsize=54:line_spacing=12:box=1:"
        "boxcolor=black@0.5:boxborderw=24:x=(w-text_w)/2:y=h-text_h-260"
    )

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", bg_path,   # loop background if shorter than audio
        "-i", voice_path,
        "-vf", vf,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",                             # stop when voiceover ends
        "-r", "30", "-movflags", "+faststart",
        out_path,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Clean up on failure before raising
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"FFmpeg failed:\n{proc.stderr[-2000:]}",
        )

    # Stream the file back and delete the temp dir afterwards
    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="short.mp4",
        background=BackgroundTask(shutil.rmtree, tmp, True),
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "shorts-auto", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
