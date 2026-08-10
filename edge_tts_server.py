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

FREE-TIER LIMITS:
  Render free tier: 512 MB RAM, shared CPU, 100 GB bandwidth/month.
  FFmpeg settings here are tuned to stay well within that envelope.
  Each input video is capped at 80 MB. Output is capped at ~60 MB.
  Errors are logged to /tmp/render_errors.log for post-mortem diagnosis.
"""

import gc
import io
import json
import logging
import os
import subprocess
import tempfile
import textwrap

import edge_tts
from fastapi import FastAPI, Form, Response, UploadFile
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
# All render errors (including full FFmpeg stderr) go here so failures are
# diagnosable without guessing.  GET /logs returns the last 8 KB of this file.
LOG_FILE = "/tmp/render_errors.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),          # also prints to Render's console log
    ],
)
log = logging.getLogger("render")

# ── FFmpeg free-tier constants ────────────────────────────────────────────────
# ultrafast  → least CPU/RAM; quality is fine for a 60-second Short
# CRF 30     → ~30-50 MB output for a 60-second clip at 1080×1920
# threads 1  → keeps RSS comfortably under 512 MB
# maxrate    → hard VBV cap so a complex scene can't spike RAM/bandwidth
PRESET      = "ultrafast"
CRF         = "30"
THREADS     = "1"
MAX_BITRATE = "900k"     # peak video bitrate cap
BUF_SIZE    = "1800k"    # VBV buffer = 2 × maxrate
AUDIO_BR    = "96k"      # voice-only track; 96k AAC is transparent enough
MAX_INPUT_MB = 80        # reject any single video clip larger than this

app = FastAPI(title="Shorts Auto", version="2.1.0")


# ── Helpers ───────────────────────────────────────────────────────────────────

def escape_line(text: str) -> str:
    """Escape a single line of text for safe use inside FFmpeg drawtext text='' value."""
    return (
        text
        .replace("\\", r"\\")   # must be first
        .replace("'",  r"\'")
        .replace(":",  r"\:")
        .replace("%",  r"\%")
    )


def caption_drawtext_filters(text: str, duration: float,
                              words_per_chunk: int = 3) -> list[str]:
    """
    Build a list of FFmpeg drawtext filter strings that show the caption
    three words at a time, each chunk appearing for an equal slice of the
    scene duration, with a semi-transparent black background box.

    Example for duration=9s, 9 words:
      chunk 0: words 1-3  shown from t=0   to t=3
      chunk 1: words 4-6  shown from t=3   to t=6
      chunk 2: words 7-9  shown from t=6   to t=9
    """
    words = text.split()
    chunks = [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, max(len(words), 1), words_per_chunk)
    ] or [text]

    chunk_dur = duration / len(chunks)
    filters = []
    for i, chunk in enumerate(chunks):
        t_start = i * chunk_dur
        t_end   = (i + 1) * chunk_dur
        escaped = escape_line(chunk)
        filters.append(
            f"drawtext=text='{escaped}'"
            f":enable='between(t,{t_start:.3f},{t_end:.3f})'"
            ":fontcolor=white"
            ":fontsize=58"
            ":box=1"
            ":boxcolor=black@0.65"
            ":boxborderw=22"
            ":x=(w-text_w)/2"
            ":y=h-text_h-220"
        )
    return filters



def run_ffmpeg(cmd: list[str], label: str) -> tuple[bool, str]:
    """
    Run an FFmpeg command, log full stderr regardless of outcome.
    Returns (success, stderr_tail).
    """
    log.info("FFmpeg [%s] start: %s", label, " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        # safety: never let a single FFmpeg call run longer than 5 minutes
        timeout=300,
    )
    stderr = result.stderr
    # Always write full stderr so any warning is visible post-mortem
    log.info("FFmpeg [%s] exit=%d stderr:\n%s", label, result.returncode, stderr)
    return result.returncode == 0, stderr


# ── TTS ───────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-GuyNeural"


@app.post("/tts")
async def generate_speech(req: TTSRequest):
    communicate = edge_tts.Communicate(req.text, req.voice)
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    audio_buffer.seek(0)
    return Response(content=audio_buffer.read(), media_type="audio/mpeg")


# ── Render ────────────────────────────────────────────────────────────────────

@app.post("/render")
async def render_short(
    audio: UploadFile,
    video1: UploadFile,
    video2: UploadFile,
    video3: UploadFile,
    video4: UploadFile,
    scenes: str = Form(...),
):
    log.info("Render request received")
    scene_data = json.loads(scenes)
    uploads = {1: video1, 2: video2, 3: video3, 4: video4}

    with tempfile.TemporaryDirectory() as tmp:

        # ── 1. Write audio ────────────────────────────────────────────────
        audio_path = os.path.join(tmp, "audio.mp3")
        audio_bytes = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        log.info("Audio written: %d bytes", len(audio_bytes))
        del audio_bytes          # free Python-side copy immediately
        gc.collect()

        processed_clips: list[str] = []

        # ── 2. Process each scene independently ───────────────────────────
        for scene in sorted(scene_data, key=lambda s: s["sceneNumber"]):
            num      = scene["sceneNumber"]
            duration = float(scene["durationSeconds"])
            log.info("Scene %d | duration=%.1fs | text=%r", num, duration, scene["text"])

            # Read → write → del: never hold more than one clip in RAM at once
            raw_path = os.path.join(tmp, f"raw{num}.mp4")
            raw_bytes = await uploads[num].read()

            mb = len(raw_bytes) / (1024 * 1024)
            if mb > MAX_INPUT_MB:
                log.error("Scene %d video too large: %.1f MB (limit %d MB)", num, mb, MAX_INPUT_MB)
                return Response(
                    content=f"Scene {num} video is {mb:.1f} MB; limit is {MAX_INPUT_MB} MB.",
                    media_type="text/plain",
                    status_code=413,
                )

            with open(raw_path, "wb") as f:
                f.write(raw_bytes)
            log.info("Scene %d raw written: %.1f MB", num, mb)
            del raw_bytes        # release before FFmpeg allocates its own buffers
            gc.collect()

            clip_path = os.path.join(tmp, f"clip{num}.mp4")

            # Build one drawtext filter per 3-word chunk, timed across the scene
            drawtext_parts = caption_drawtext_filters(scene["text"], duration)
            vf = (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,"
                + ",".join(drawtext_parts)
            )

            # Encode this scene clip (no audio; muxed at the end)
            ok, stderr = run_ffmpeg([
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", raw_path,
                "-t", str(duration),
                "-vf", vf,
                "-an",
                "-c:v", "libx264",
                "-preset", PRESET,
                "-crf", CRF,
                "-maxrate", MAX_BITRATE,
                "-bufsize", BUF_SIZE,
                "-threads", THREADS,
                "-r", "30",
                clip_path,
            ], label=f"scene{num}")

            # Raw clip no longer needed; delete it to free disk space
            os.remove(raw_path)

            if not ok:
                return Response(
                    content=f"FFmpeg failed on scene {num}. Last stderr:\n{stderr[-3000:]}",
                    media_type="text/plain",
                    status_code=500,
                )

            processed_clips.append(clip_path)
            log.info("Scene %d encoded → %s", num, clip_path)
            gc.collect()      # nudge GC between scenes

        # ── 3. Concatenate scene clips ────────────────────────────────────
        concat_list = os.path.join(tmp, "concat.txt")
        with open(concat_list, "w") as f:
            for clip in processed_clips:
                f.write(f"file '{clip}'\n")

        concatenated = os.path.join(tmp, "concatenated.mp4")
        ok, stderr = run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy",
            concatenated,
        ], label="concat")

        # Individual clips no longer needed after concat
        for clip in processed_clips:
            try:
                os.remove(clip)
            except OSError:
                pass

        if not ok:
            return Response(
                content=f"FFmpeg concat failed. Last stderr:\n{stderr[-3000:]}",
                media_type="text/plain",
                status_code=500,
            )
        log.info("Concat done → %s", concatenated)

        # ── 4. Mux voiceover onto concatenated video ──────────────────────
        output_path = os.path.join(tmp, "output.mp4")
        ok, stderr = run_ffmpeg([
            "ffmpeg", "-y",
            "-i", concatenated,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",                 # video already encoded; just remux
            "-c:a", "aac", "-b:a", AUDIO_BR,
            "-shortest",
            "-movflags", "+faststart",      # web-friendly atom order
            output_path,
        ], label="mux")

        os.remove(concatenated)             # free disk before reading output

        if not ok:
            return Response(
                content=f"FFmpeg mux failed. Last stderr:\n{stderr[-3000:]}",
                media_type="text/plain",
                status_code=500,
            )

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        log.info("Render complete. Output size: %.1f MB", len(video_bytes) / (1024 * 1024))
        return Response(content=video_bytes, media_type="video/mp4")


# ── Diagnostics ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ffmpeg-render-wrapper-v2-multiscene", "version": "2.1.0"}


@app.get("/logs")
async def get_logs():
    """Return the last 8 KB of the render error log for remote diagnosis."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read()
        return Response(content=tail, media_type="text/plain")
    except FileNotFoundError:
        return Response(content="Log file not yet created.", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
