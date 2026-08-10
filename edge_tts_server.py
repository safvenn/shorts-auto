"""
Free Edge-TTS + Render HTTP wrapper for n8n - v2 (multi-scene)
----------------------------------------------------------------
Provides two endpoints:
  POST /tts    – generate MP3 voiceover via Microsoft Edge TTS (free, no key)
  POST /render – multi-scene render: 4 video clips + 1 voiceover + scenes JSON
                 → single vertical Short with per-scene captions, synced to audio.

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
    Fields:
      audio   – binary  full voiceover track
      video1  – binary  scene 1 footage
      video2  – binary  scene 2 footage
      video3  – binary  scene 3 footage
      video4  – binary  scene 4 footage
      scenes  – text    JSON: [{"sceneNumber":1,"text":"...","durationSeconds":7}, ...]
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
import json
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


def escape_drawtext(text: str) -> str:
    """Escape characters that break the FFmpeg drawtext filter."""
    return text.replace("\\", r"\\\\").replace("'", r"\'").replace(":", r"\:")


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
    video1: UploadFile,
    video2: UploadFile,
    video3: UploadFile,
    video4: UploadFile,
    scenes: str = Form(...),  # JSON string
):
    scene_data = json.loads(scenes)
    videos = {1: video1, 2: video2, 3: video3, 4: video4}

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        processed_clips = []

        for scene in sorted(scene_data, key=lambda s: s["sceneNumber"]):
            num = scene["sceneNumber"]
            duration = scene["durationSeconds"]
            caption_text = escape_drawtext(wrap_caption(scene["text"]))

            raw_path = os.path.join(tmp, f"raw{num}.mp4")
            with open(raw_path, "wb") as f:
                f.write(await videos[num].read())

            clip_path = os.path.join(tmp, f"clip{num}.mp4")

            vf = (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,"
                f"drawtext=text='{caption_text}':fontcolor=white:fontsize=48:"
                "borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-380:line_spacing=10"
            )

            # Trim/loop this scene's footage to exactly its duration,
            # scale to vertical, burn this scene's caption. No audio here -
            # audio comes from the single full voiceover track muxed at the end.
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", raw_path,
                "-t", str(duration),
                "-vf", vf,
                "-an",
                "-c:v", "libx264", "-preset", "veryfast",
                clip_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return Response(
                    content=f"FFmpeg failed on scene {num}:\n{result.stderr[-2000:]}",
                    media_type="text/plain",
                    status_code=500,
                )
            processed_clips.append(clip_path)

        # Concat the 4 processed scene clips end-to-end
        concat_list_path = os.path.join(tmp, "concat.txt")
        with open(concat_list_path, "w") as f:
            for clip in processed_clips:
                f.write(f"file '{clip}'\n")

        concatenated_path = os.path.join(tmp, "concatenated.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy",
            concatenated_path,
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return Response(
                content=f"FFmpeg concat failed:\n{result.stderr[-2000:]}",
                media_type="text/plain",
                status_code=500,
            )

        # Mux the concatenated video with the full voiceover track
        output_path = os.path.join(tmp, "output.mp4")
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", concatenated_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(mux_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return Response(
                content=f"FFmpeg mux failed:\n{result.stderr[-2000:]}",
                media_type="text/plain",
                status_code=500,
            )

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return Response(content=video_bytes, media_type="video/mp4")


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ffmpeg-render-wrapper-v2-multiscene"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
