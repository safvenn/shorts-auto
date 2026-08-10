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
import subprocess
import tempfile
import textwrap

import edge_tts
from fastapi import FastAPI, Form, Response, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Shorts Auto", version="2.0.0")


def wrap_caption(text: str, width: int = 28) -> str:
    """Break caption into short lines so it fits a 1080px-wide vertical video."""
    lines = textwrap.wrap(text, width=width)
    return r"\N".join(lines)  # \N = forced newline inside FFmpeg drawtext


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
async def render_short(
    audio: UploadFile,
    video: UploadFile,
    caption: str = Form(""),
):
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.mp3")
        video_path = os.path.join(tmp, "video.mp4")
        output_path = os.path.join(tmp, "output.mp4")

        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        with open(video_path, "wb") as f:
            f.write(await video.read())

        caption_text = wrap_caption(caption).replace("'", r"\'").replace(":", r"\:")

        # Build the filter: crop/scale footage to 1080x1920 vertical,
        # loop it if shorter than audio, burn caption near the bottom,
        # then mux with the voiceover track.
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            f"drawtext=text='{caption_text}':fontcolor=white:fontsize=48:"
            "borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-380:line_spacing=10"
        )

        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_path,   # loop background video
            "-i", audio_path,                          # voiceover (sets duration)
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",                                # cut to audio length
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return Response(
                content=f"FFmpeg failed:\n{result.stderr[-2000:]}",
                media_type="text/plain",
                status_code=500,
            )

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return Response(content=video_bytes, media_type="video/mp4")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "shorts-auto", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
