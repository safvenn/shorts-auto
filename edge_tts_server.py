"""
Free Edge-TTS + Render HTTP wrapper for n8n - v5 (drawtext captions, 4K-safe)
-------------------------------------------------------------------------------
Provides two endpoints:
  POST /tts    – generate MP3 voiceover via Microsoft Edge TTS (free, no key)
  POST /render – multi-scene render: 4 video clips + 1 voiceover + scenes JSON
                 → single vertical Short with:
                   • FFmpeg drawtext captions (single-stream, OOM-safe on 4K input)
                   • Bold yellow text (#FFE600), thick black border, drop shadow
                   • Auto emoji injection (keyword → 🔥💰★🏆⚡🧠😂😱🔑❤)
                   • 2-word chunks, exact timing via enable='between(t,...)'
                   • Captions at lower-third (430px above bottom)
                   • CRF 23, ultrafast preset — safe on 512 MB AWS free-tier

SETUP:
  Docker:
    docker build -t shorts-auto . && docker run -p 8000:8000 shorts-auto

  Local (TTS-only):
    pip install fastapi uvicorn edge-tts python-multipart
    python edge_tts_server.py

USAGE:
  POST /tts   { "text": "...", "voice": "en-US-GuyNeural" }  → MP3
  POST /render  (multipart/form-data)
    audio, video1-4, scenes JSON  → MP4
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

# ── Logging ───────────────────────────────────────────────────────────────────
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

# ── FFmpeg quality settings ───────────────────────────────────────────────────
# ultrafast + CRF 23: safe on 512 MB free-tier while decoding 4K input.
# The encoder only touches the 1080×1920 scaled frame, not the 4K original,
# so CRF 23 ultrafast still produces clean-looking vertical Shorts output.
PRESET       = "ultrafast"
CRF          = "23"
THREADS      = "1"
MAX_BITRATE  = "4000k"
BUF_SIZE     = "8000k"
AUDIO_BR     = "128k"
MAX_INPUT_MB = 80

# ── Caption layout ────────────────────────────────────────────────────────────
WORDS_PER_CHUNK   = 2     # words per caption card
CAPTION_FONT_SIZE = 82    # drawtext fontsize
CAPTION_BOTTOM_PAD= 430   # px from bottom of frame

# ── Emoji keyword map (appended as plain text after each chunk) ────────────────
# drawtext renders these as monochrome glyphs or boxes — acceptable fallback.
EMOJI_MAP: dict[str, str] = {
    "fire": "🔥", "hot": "🔥", "burn": "🔥", "heat": "🔥", "flame": "🔥",
    "money": "💰", "cash": "💰", "rich": "💰", "earn": "💰", "profit": "💰",
    "income": "💰", "dollar": "💰", "wealth": "💰",
    "love": "❤",  "heart": "❤",  "care": "❤",  "feel": "❤",
    "star": "★",  "best": "★",   "amazing": "★", "great": "★", "awesome": "★",
    "top": "★",
    "win": "🏆", "winner": "🏆", "victory": "🏆", "champion": "🏆",
    "succeed": "🏆", "success": "🏆",
    "fast": "⚡", "speed": "⚡", "quick": "⚡", "boost": "⚡", "power": "⚡",
    "mind": "🧠", "brain": "🧠", "think": "🧠", "smart": "🧠", "learn": "🧠",
    "secret": "🔑", "key": "🔑", "unlock": "🔑",
    "laugh": "😂", "funny": "😂", "joke": "😂",
    "shock": "😱", "wow": "😱", "crazy": "😱", "insane": "😱", "wild": "😱",
    "grow": "🚀", "growth": "🚀", "launch": "🚀",
    "work": "💪", "grind": "💪", "hustle": "💪", "strong": "💪",
}

app = FastAPI(title="Shorts Auto", version="5.0.0")


# ── Font helpers ──────────────────────────────────────────────────────────────

_BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _find_font(candidates: list[str]) -> str:
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


# ── Caption via FFmpeg drawtext (single-stream, OOM-safe on 4K input) ─────────

def _esc(text: str) -> str:
    """Escape text for FFmpeg drawtext filter (colon, backslash, apostrophe)."""
    return text.replace("\\", "\\\\").replace("'", "\\\\\\'").replace(":", "\\:")


def _emoji_for_chunk(chunk: str) -> str:
    """Return chunk text + appended emoji char if a keyword matches."""
    for word in chunk.lower().split():
        clean = word.strip(".,!?;:")
        if clean in EMOJI_MAP:
            return chunk + " " + EMOJI_MAP[clean]
    return chunk


def caption_drawtext_filters(
    text: str,
    duration: float,
    font_path: str,
) -> list[str]:
    """
    Build a list of FFmpeg drawtext filter strings — one per 2-word caption chunk.

    Styling:
    - Yellow (#FFE600) bold text centred horizontally
    - Black border (bordercolor=black, borderw=5)
    - Positioned CAPTION_BOTTOM_PAD px from the bottom of the frame
    - Each chunk shown for chunk_dur seconds via 'enable=between(t,...)'
    """
    words = text.split()
    if not words:
        return []

    chunks = [
        " ".join(words[i: i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]
    chunk_dur = duration / max(len(chunks), 1)
    filters   = []

    font_arg = f":fontfile={font_path}" if font_path else ""

    for i, chunk in enumerate(chunks):
        t_start = round(i * chunk_dur, 4)
        t_end   = round((i + 1) * chunk_dur, 4)
        label   = _emoji_for_chunk(chunk)

        # Shadow pass (offset by 4px, dark semi-transparent)
        shadow = (
            f"drawtext=text='{_esc(label)}'{font_arg}"
            f":fontsize={CAPTION_FONT_SIZE}"
            f":fontcolor=black@0.6"
            f":x=(w-text_w)/2+4:y=h-text_h-{CAPTION_BOTTOM_PAD}+4"
            f":borderw=0"
            f":enable='between(t,{t_start},{t_end})'"
        )
        # Main text pass (yellow + black border)
        main = (
            f"drawtext=text='{_esc(label)}'{font_arg}"
            f":fontsize={CAPTION_FONT_SIZE}"
            f":fontcolor=#FFE600"
            f":x=(w-text_w)/2:y=h-text_h-{CAPTION_BOTTOM_PAD}"
            f":bordercolor=black:borderw=5"
            f":enable='between(t,{t_start},{t_end})'"
        )
        filters += [shadow, main]
        log.info("Caption [%.3f-%.3f] %r", t_start, t_end, label)

    return filters


# ── FFmpeg wrapper ────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: list[str], label: str) -> tuple[bool, str]:
    """Run FFmpeg, log stderr. Returns (success, stderr)."""
    log.info("FFmpeg [%s] start: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log.info("FFmpeg [%s] exit=%d stderr:\n%s", label, result.returncode, result.stderr)
    return result.returncode == 0, result.stderr


# ── TTS ───────────────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-GuyNeural"


@app.post("/tts")
async def generate_speech(req: TTSRequest):
    communicate = edge_tts.Communicate(req.text, req.voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/mpeg")


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
    log.info("Render request received (v5 drawtext captions, single-stream, 4K-safe)")
    scene_data = json.loads(scenes)
    uploads    = {1: video1, 2: video2, 3: video3, 4: video4}
    bold_font  = _find_font(_BOLD_FONT_CANDIDATES)
    log.info("Caption font: %s", bold_font or "(system default)")

    with tempfile.TemporaryDirectory() as tmp:

        # ── 1. Write audio ─────────────────────────────────────────────────
        audio_path  = os.path.join(tmp, "audio.mp3")
        audio_bytes = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        log.info("Audio written: %d bytes", len(audio_bytes))
        del audio_bytes
        gc.collect()

        processed_clips: list[str]   = []
        scene_durations: list[float] = []

        # ── 2. Process each scene ──────────────────────────────────────────
        for scene in sorted(scene_data, key=lambda s: s["sceneNumber"]):
            num      = scene["sceneNumber"]
            duration = float(scene["durationSeconds"])
            log.info("Scene %d | %.1fs | %r", num, duration, scene["text"])

            # Write raw video
            raw_path  = os.path.join(tmp, f"raw{num}.mp4")
            raw_bytes = await uploads[num].read()
            mb = len(raw_bytes) / (1024 * 1024)
            if mb > MAX_INPUT_MB:
                return Response(
                    content=f"Scene {num} video is {mb:.1f} MB; limit {MAX_INPUT_MB} MB.",
                    media_type="text/plain", status_code=413,
                )
            with open(raw_path, "wb") as f:
                f.write(raw_bytes)
            log.info("Scene %d raw: %.1f MB", num, mb)
            del raw_bytes
            gc.collect()

            clip_path = os.path.join(tmp, f"clip{num}.mp4")

            # ── Build -vf chain: scale + crop + drawtext captions ───────────
            # Single-stream approach: decode 4K → scale to 1080p → draw text.
            # No second stream = no overlay filter = no OOM on 512 MB RAM.
            cap_filters = caption_drawtext_filters(scene["text"], duration, bold_font)
            vf_chain = ",".join(
                ["scale=1080:1920:force_original_aspect_ratio=increase", "crop=1080:1920"]
                + cap_filters
            )

            ok, stderr = run_ffmpeg([
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-t", str(duration), "-i", raw_path,
                "-vf", vf_chain,
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

            # Clean up raw video immediately to free disk space
            os.remove(raw_path)

            if not ok:
                return Response(
                    content=f"FFmpeg failed on scene {num}:\n{stderr[-3000:]}",
                    media_type="text/plain", status_code=500,
                )

            processed_clips.append(clip_path)
            scene_durations.append(duration)
            log.info("Scene %d encoded → %s", num, clip_path)
            gc.collect()

        # ── 3. Concat clips (stream-copy, zero RAM) ────────────────────────
        n = len(processed_clips)
        if n == 1:
            concatenated = processed_clips[0]
        else:
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

            for clip in processed_clips:
                try:
                    os.remove(clip)
                except OSError:
                    pass

            if not ok:
                return Response(
                    content=f"FFmpeg concat failed:\n{stderr[-3000:]}",
                    media_type="text/plain", status_code=500,
                )

        # ── 4. Mux voiceover ───────────────────────────────────────────────
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

        if n > 1:
            try:
                os.remove(concatenated)
            except OSError:
                pass

        if not ok:
            return Response(
                content=f"FFmpeg mux failed:\n{stderr[-3000:]}",
                media_type="text/plain", status_code=500,
            )

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        log.info("Render complete (v4). %.1f MB", len(video_bytes) / (1024 * 1024))
        return Response(content=video_bytes, media_type="video/mp4")


# ── Diagnostics ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    bold  = _find_font(_BOLD_FONT_CANDIDATES)
    emoji = _find_font(_EMOJI_FONT_CANDIDATES)
    return {
        "status": "ok",
        "service": "ffmpeg-render-wrapper-v4-pillow-concat",
        "version": "4.1.0",
        "bold_font":  bold  or "PIL default",
        "emoji_font": emoji or "not found",
        "crf": CRF, "preset": PRESET,
        "caption_bottom_pad_px": CAPTION_BOTTOM_PAD,
    }


@app.get("/logs")
async def get_logs():
    """Last 8 KB of render error log."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            return Response(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        return Response(content="Log file not yet created.", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
