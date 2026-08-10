"""
Free Edge-TTS + Render HTTP wrapper for n8n - v3 (cinematic quality)
---------------------------------------------------------------------
Provides two endpoints:
  POST /tts    – generate MP3 voiceover via Microsoft Edge TTS (free, no key)
  POST /render – multi-scene render: 4 video clips + 1 voiceover + scenes JSON
                 → single vertical Short with HIGH-QUALITY:
                   • Bold yellow top captions (karaoke-style, 2 words/chunk)
                   • Auto emoji injection (keyword → 🔥💰⭐🏆⚡🧠😂😱🔑❤️)
                   • Per-scene Ken Burns zoom/pan effect
                   • xfade transitions between scenes (fade/slideleft/wipeleft)
                   • Cinema vignette overlay per scene
                   • Caption pop-in bounce (large → normal size within 0.15s)

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

import edge_tts
from fastapi import FastAPI, Form, Response, UploadFile
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = "/tmp/render_errors.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("render")

# ── FFmpeg free-tier constants ────────────────────────────────────────────────
PRESET       = "ultrafast"
CRF          = "28"           # slightly better quality than v2
THREADS      = "1"
MAX_BITRATE  = "900k"
BUF_SIZE     = "1800k"
AUDIO_BR     = "96k"
MAX_INPUT_MB = 80

# ── Caption constants ─────────────────────────────────────────────────────────
WORDS_PER_CHUNK     = 2      # 2 words at a time — punchy Short pacing
CAPTION_Y           = "120"  # distance from top in pixels
CAPTION_FONTSIZE    = 72     # base font size
CAPTION_BOUNCE_SIZE = 92     # oversized during bounce-in
CAPTION_BOUNCE_DUR  = 0.15   # seconds the pop/bounce lasts
TRANSITION_DUR      = 0.4    # xfade duration between scenes (seconds)

# Cycle of xfade transition names (FFmpeg built-ins)
TRANSITIONS = ["fade", "slideleft", "wipeleft", "zoomin"]

# ── Emoji keyword map ─────────────────────────────────────────────────────────
EMOJI_MAP: dict[str, str] = {
    "fire": "🔥", "hot": "🔥", "burn": "🔥", "heat": "🔥", "flame": "🔥",
    "money": "💰", "cash": "💰", "rich": "💰", "earn": "💰", "profit": "💰",
    "income": "💰", "dollar": "💰", "wealth": "💰",
    "love": "❤️", "heart": "❤️", "care": "❤️", "feel": "❤️",
    "star": "⭐", "best": "⭐", "amazing": "⭐", "great": "⭐", "awesome": "⭐",
    "top": "⭐",
    "win": "🏆", "winner": "🏆", "victory": "🏆", "champion": "🏆",
    "succeed": "🏆", "success": "🏆",
    "fast": "⚡", "speed": "⚡", "quick": "⚡", "boost": "⚡", "power": "⚡",
    "mind": "🧠", "brain": "🧠", "think": "🧠", "smart": "🧠", "learn": "🧠",
    "secret": "🔑", "key": "🔑", "unlock": "🔑",
    "laugh": "😂", "funny": "😂", "joke": "😂", "lol": "😂",
    "shock": "😱", "wow": "😱", "crazy": "😱", "insane": "😱", "wild": "😱",
    "grow": "🚀", "growth": "🚀", "launch": "🚀", "go": "🚀", "start": "🚀",
    "life": "🌟", "new": "🌟", "world": "🌍",
    "work": "💪", "grind": "💪", "hustle": "💪", "strong": "💪",
}

app = FastAPI(title="Shorts Auto", version="3.0.0")


# ── Helpers ───────────────────────────────────────────────────────────────────

def escape_drawtext(text: str) -> str:
    """Escape a string for safe use inside FFmpeg drawtext text='' value."""
    return (
        text
        .replace("\\", r"\\")   # must be first
        .replace("'",  r"\'")
        .replace(":",  r"\:")
        .replace(",",  r"\,")
        .replace("%",  r"\%")
    )


def emoji_for_chunk(chunk: str) -> str:
    """
    Return a single emoji to append to a caption chunk based on keyword lookup.
    Returns "" if no match.
    """
    for word in chunk.lower().split():
        clean = word.strip(".,!?;:")
        if clean in EMOJI_MAP:
            return " " + EMOJI_MAP[clean]
    return ""


def build_font_path() -> str:
    """
    Return the best available bold font path on the system.
    Priority: Impact → DejaVuSans-Bold → any DejaVu → empty (FFmpeg default).
    """
    candidates = [
        # Liberation Sans Bold (installed via fonts-liberation)
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        # DejaVu Bold (always present in our Docker image)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        # Fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            log.info("Caption font: %s", path)
            return path
    log.warning("No preferred font found; FFmpeg will use its default.")
    return ""


def caption_drawtext_filters(
    text: str,
    duration: float,
    font_path: str,
) -> list[str]:
    """
    Build FFmpeg drawtext filter strings for animated top captions.

    Features per chunk:
    - 2-word chunks, evenly distributed across scene duration
    - Bold yellow text with thick black border (stroke) + drop shadow
    - Auto emoji appended based on keyword map
    - Caption bounce: oversized text for first CAPTION_BOUNCE_DUR seconds,
      then normal size for the rest of the chunk duration
    - Positioned at the top of the frame (y = CAPTION_Y)
    """
    words = text.split()
    if not words:
        words = [text or " "]

    chunks = [
        " ".join(words[i: i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]

    chunk_dur = duration / max(len(chunks), 1)
    filters: list[str] = []
    font_arg = f":fontfile='{font_path}'" if font_path else ""

    for i, chunk in enumerate(chunks):
        t_start    = i * chunk_dur
        t_end      = (i + 1) * chunk_dur
        bounce_end = min(t_start + CAPTION_BOUNCE_DUR, t_end)

        display_text = chunk + emoji_for_chunk(chunk)
        escaped      = escape_drawtext(display_text)

        enable_bounce = f"between(t\\,{t_start:.3f}\\,{bounce_end:.3f})"
        enable_normal = f"between(t\\,{bounce_end:.3f}\\,{t_end:.3f})"

        common_style = (
            f":fontcolor=yellow"
            f":bordercolor=black:borderw=6"
            f":shadowcolor=black@0.8:shadowx=4:shadowy=4"
            f":x=(w-text_w)/2"
            f":y={CAPTION_Y}"
            f"{font_arg}"
        )

        # Bounce frame: oversized pop
        filters.append(
            f"drawtext=text='{escaped}'"
            f":enable='{enable_bounce}'"
            f":fontsize={CAPTION_BOUNCE_SIZE}"
            + common_style
        )
        # Normal frame
        filters.append(
            f"drawtext=text='{escaped}'"
            f":enable='{enable_normal}'"
            f":fontsize={CAPTION_FONTSIZE}"
            + common_style
        )

    return filters


def ken_burns_filter(scene_num: int, duration: float) -> str:
    """
    Return a zoompan filter string that slowly zooms + pans the frame,
    giving a cinematic Ken Burns effect.

    Direction alternates:
    - Odd scenes:  zoom in (1.0→1.05), drift toward bottom-right
    - Even scenes: zoom out (1.05→1.0), drift toward top-left
    """
    fps          = 30
    total_frames = max(int(duration * fps), 1)

    if scene_num % 2 == 1:
        zoom_expr = f"1.00+0.05*on/{total_frames}"
        x_expr    = f"iw/2-(iw/zoom/2)+iw*0.02*(on/{total_frames})"
        y_expr    = f"ih/2-(ih/zoom/2)+ih*0.02*(on/{total_frames})"
    else:
        zoom_expr = f"1.05-0.05*on/{total_frames}"
        x_expr    = f"iw/2-(iw/zoom/2)+iw*0.02*(1-on/{total_frames})"
        y_expr    = f"ih/2-(ih/zoom/2)+ih*0.02*(1-on/{total_frames})"

    return (
        f"zoompan=z='{zoom_expr}'"
        f":x='{x_expr}'"
        f":y='{y_expr}'"
        f":d={total_frames}"
        f":s=1080x1920"
        f":fps={fps}"
    )


def vignette_filter() -> str:
    """Cinema edge-darkening vignette."""
    return "vignette=angle=PI/4:mode=backward"


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
        timeout=300,
    )
    stderr = result.stderr
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
    log.info("Render request received (v3 cinematic)")
    scene_data = json.loads(scenes)
    uploads    = {1: video1, 2: video2, 3: video3, 4: video4}
    font_path  = build_font_path()

    with tempfile.TemporaryDirectory() as tmp:

        # ── 1. Write audio ────────────────────────────────────────────────
        audio_path  = os.path.join(tmp, "audio.mp3")
        audio_bytes = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        log.info("Audio written: %d bytes", len(audio_bytes))
        del audio_bytes
        gc.collect()

        processed_clips: list[str]  = []
        scene_durations: list[float] = []

        # ── 2. Process each scene independently ───────────────────────────
        for scene in sorted(scene_data, key=lambda s: s["sceneNumber"]):
            num      = scene["sceneNumber"]
            duration = float(scene["durationSeconds"])
            log.info("Scene %d | duration=%.1fs | text=%r", num, duration, scene["text"])

            raw_path  = os.path.join(tmp, f"raw{num}.mp4")
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
            del raw_bytes
            gc.collect()

            clip_path = os.path.join(tmp, f"clip{num}.mp4")

            # ── Build filter chain ────────────────────────────────────────
            # Step 1: scale + crop to 1080×1920
            scale_crop = (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920"
            )
            # Step 2: Ken Burns zoom/pan
            kb = ken_burns_filter(num, duration)
            # Step 3: Vignette
            vignette = vignette_filter()
            # Step 4: Captions (bold yellow, top, bounce-in, emoji)
            captions = caption_drawtext_filters(scene["text"], duration, font_path)

            vf = ",".join([scale_crop, kb, vignette] + captions)

            # Render slightly longer than duration to allow xfade overlap
            render_duration = duration + TRANSITION_DUR

            ok, stderr = run_ffmpeg([
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", raw_path,
                "-t", str(render_duration),
                "-vf", vf,
                "-an",
                "-c:v", "libx264",
                "-preset", PRESET,
                "-crf", CRF,
                "-maxrate", MAX_BITRATE,
                "-bufsize", BUF_SIZE,
                "-threads", THREADS,
                "-r", "30",
                "-pix_fmt", "yuv420p",
                clip_path,
            ], label=f"scene{num}")

            os.remove(raw_path)

            if not ok:
                return Response(
                    content=f"FFmpeg failed on scene {num}. Last stderr:\n{stderr[-3000:]}",
                    media_type="text/plain",
                    status_code=500,
                )

            processed_clips.append(clip_path)
            scene_durations.append(duration)
            log.info("Scene %d encoded → %s", num, clip_path)
            gc.collect()

        # ── 3. Build xfade transition chain ───────────────────────────────
        #
        # For N clips we need N-1 xfade filters chained together.
        # Each xfade offset = cumulative duration of all prior scenes minus
        # transition overlaps already consumed.
        #
        n = len(processed_clips)

        if n == 1:
            concatenated = processed_clips[0]
            log.info("Single scene – skipping xfade")
        else:
            input_args: list[str] = []
            for clip in processed_clips:
                input_args += ["-i", clip]

            filter_parts: list[str] = []
            cumulative_offset = 0.0
            prev_label = "[0:v]"

            for i in range(1, n):
                cumulative_offset += scene_durations[i - 1] - (TRANSITION_DUR if i > 1 else 0)
                offset     = max(cumulative_offset - TRANSITION_DUR, 0)
                transition = TRANSITIONS[(i - 1) % len(TRANSITIONS)]
                out_label  = f"[x{i}]" if i < n - 1 else "[vout]"

                filter_parts.append(
                    f"{prev_label}[{i}:v]"
                    f"xfade=transition={transition}"
                    f":duration={TRANSITION_DUR}"
                    f":offset={offset:.3f}"
                    f"{out_label}"
                )
                prev_label = out_label

            filter_complex = "; ".join(filter_parts)
            concatenated   = os.path.join(tmp, "concatenated.mp4")

            ok, stderr = run_ffmpeg(
                ["ffmpeg", "-y"]
                + input_args
                + [
                    "-filter_complex", filter_complex,
                    "-map", "[vout]",
                    "-c:v", "libx264",
                    "-preset", PRESET,
                    "-crf", CRF,
                    "-maxrate", MAX_BITRATE,
                    "-bufsize", BUF_SIZE,
                    "-threads", THREADS,
                    "-r", "30",
                    "-pix_fmt", "yuv420p",
                    concatenated,
                ],
                label="xfade",
            )

            for clip in processed_clips:
                try:
                    os.remove(clip)
                except OSError:
                    pass

            if not ok:
                return Response(
                    content=f"FFmpeg xfade failed. Last stderr:\n{stderr[-3000:]}",
                    media_type="text/plain",
                    status_code=500,
                )
            log.info("xfade concat done → %s", concatenated)

        # ── 4. Mux voiceover onto video ───────────────────────────────────
        output_path = os.path.join(tmp, "output.mp4")
        ok, stderr = run_ffmpeg([
            "ffmpeg", "-y",
            "-i", concatenated,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", AUDIO_BR,
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ], label="mux")

        # Clean up concat file if it was a temp file (multi-scene)
        if n > 1:
            try:
                os.remove(concatenated)
            except OSError:
                pass

        if not ok:
            return Response(
                content=f"FFmpeg mux failed. Last stderr:\n{stderr[-3000:]}",
                media_type="text/plain",
                status_code=500,
            )

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        log.info("Render complete (v3). Output size: %.1f MB", len(video_bytes) / (1024 * 1024))
        return Response(content=video_bytes, media_type="video/mp4")


# ── Diagnostics ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    font = build_font_path()
    return {
        "status": "ok",
        "service": "ffmpeg-render-wrapper-v3-cinematic",
        "version": "3.0.0",
        "caption_font": font or "ffmpeg-default",
        "transitions": TRANSITIONS,
    }


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
