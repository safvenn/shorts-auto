"""
Free Edge-TTS + Render HTTP wrapper for n8n - v4 (Pillow captions + High Quality)
---------------------------------------------------------------------------------
Provides two endpoints:
  POST /tts    – generate MP3 voiceover via Microsoft Edge TTS (free, no key)
  POST /render – multi-scene render: 4 video clips + 1 voiceover + scenes JSON
                 → single vertical Short with:
                   • Pillow-rendered caption PNGs (true color emoji, no FFmpeg drawtext)
                   • Bold yellow text, thick black stroke, drop shadow
                   • Auto emoji injection (keyword → 🔥💰⭐🏆⚡🧠😂😱🔑❤️)
                   • 2-word chunks, zero-lag exact timing via FFmpeg overlay
                   • Captions at lower-third (430px above bottom, below screen center)
                   • High quality encode: CRF 18, fast preset, 4 Mbps cap

SETUP:
  Docker:
    docker build -t shorts-auto . && docker run -p 8000:8000 shorts-auto

  Local (TTS-only):
    pip install fastapi uvicorn edge-tts python-multipart Pillow
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
from PIL import Image, ImageDraw, ImageFont
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
# CRF 18 = high quality (visually near-lossless).
# preset fast = crisp encoding quality
# 4 Mbps cap keeps file sizes sharp while reasonable
PRESET       = "fast"
CRF          = "18"
THREADS      = "1"
MAX_BITRATE  = "4000k"
BUF_SIZE     = "8000k"
AUDIO_BR     = "128k"
MAX_INPUT_MB = 80

# ── Caption layout ────────────────────────────────────────────────────────────
WORDS_PER_CHUNK      = 2      # 2 words per caption card
VIDEO_W              = 1080
VIDEO_H              = 1920
CAPTION_FONT_SIZE    = 82     # large, punchy
CAPTION_STROKE_W     = 5      # black border thickness (px)
CAPTION_SHADOW_OFF   = 5      # drop shadow offset (px)
CAPTION_BOTTOM_PAD   = 430    # px from bottom of frame to bottom of caption
# CRITICAL: ALL caption PNGs must be EXACTLY this size.
# The concat demuxer feeds them into one overlay filter; any size change
# causes FFmpeg to "Reconfigure filter graph" and produce 0 frames.
CAPTION_PNG_W        = VIDEO_W - 20   # 1060px — fixed width
CAPTION_PNG_H        = 150            # fixed height (fits font+stroke+shadow)

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

app = FastAPI(title="Shorts Auto", version="4.0.0")


# ── Font helpers ──────────────────────────────────────────────────────────────

_BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

_EMOJI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
]


def _find_font(candidates: list[str]) -> str:
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    return ImageFont.load_default()


# ── Caption rendering (Pillow) ────────────────────────────────────────────────

def emoji_for_chunk(chunk: str) -> tuple[str, str]:
    """
    Return (clean_chunk_text, emoji_str).
    If a keyword matches, returns (chunk, emoji). Otherwise returns (chunk, "").
    """
    for word in chunk.lower().split():
        clean = word.strip(".,!?;:")
        if clean in EMOJI_MAP:
            return chunk, " " + EMOJI_MAP[clean]
    return chunk, ""


def render_caption_png(
    chunk_text: str,
    emoji_str: str,
    bold_font_path: str,
    emoji_font_path: str,
    font_size: int = CAPTION_FONT_SIZE,
) -> Image.Image:
    """
    Render caption as a FIXED-SIZE (CAPTION_PNG_W x CAPTION_PNG_H) transparent RGBA PNG.

    IMPORTANT: Fixed size is critical — the concat demuxer feeds all caption PNGs
    into a single overlay filter. If any PNG differs in size, FFmpeg reconfigures
    the filter graph and produces 0 output frames (silent crash).

    All text/emoji is centered in this fixed canvas.
    """
    bold_font  = _load_font(bold_font_path, font_size)
    emoji_font = _load_font(emoji_font_path, font_size) if emoji_font_path else None

    # ── Fixed canvas — MUST match CAPTION_PNG_W × CAPTION_PNG_H exactly ──────
    img  = Image.new("RGBA", (CAPTION_PNG_W, CAPTION_PNG_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Measure text on a probe canvas
    probe      = Image.new("RGBA", (CAPTION_PNG_W * 2, CAPTION_PNG_H * 4), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)

    bbox_text = probe_draw.textbbox(
        (0, 0), chunk_text, font=bold_font, stroke_width=CAPTION_STROKE_W
    )
    text_w = bbox_text[2] - bbox_text[0]
    text_h = bbox_text[3] - bbox_text[1]

    # Measure emoji width if present
    emoji_w = 0
    ef = emoji_font if emoji_font else bold_font
    if emoji_str:
        bbox_em = probe_draw.textbbox((0, 0), emoji_str, font=ef)
        emoji_w = bbox_em[2] - bbox_em[0] + 8  # 8px gap

    total_w = text_w + emoji_w

    # Center the whole group horizontally, center text vertically
    x = (CAPTION_PNG_W - total_w) // 2 - bbox_text[0]
    y = (CAPTION_PNG_H - text_h) // 2 - bbox_text[1]

    # 1. Drop shadow
    draw.text(
        (x + CAPTION_SHADOW_OFF, y + CAPTION_SHADOW_OFF),
        chunk_text, font=bold_font, fill=(0, 0, 0, 170),
    )

    # 2. Main text (yellow with black stroke)
    draw.text(
        (x, y), chunk_text, font=bold_font,
        fill=(255, 230, 0, 255),
        stroke_width=CAPTION_STROKE_W,
        stroke_fill=(0, 0, 0, 255),
    )

    # 3. Emoji (right of text)
    if emoji_str:
        x_emoji = x + text_w + bbox_text[0] + 8
        try:
            if emoji_font:
                draw.text((x_emoji, y), emoji_str, font=ef, embedded_color=True)
            else:
                draw.text(
                    (x_emoji, y), emoji_str, font=ef,
                    fill=(255, 230, 0, 255),
                    stroke_width=CAPTION_STROKE_W,
                    stroke_fill=(0, 0, 0, 255),
                )
        except Exception:
            draw.text(
                (x_emoji, y), emoji_str, font=bold_font,
                fill=(255, 230, 0, 255),
                stroke_width=CAPTION_STROKE_W,
                stroke_fill=(0, 0, 0, 255),
            )

    return img


def make_caption_overlays(
    text: str,
    duration: float,
    bold_font_path: str,
    emoji_font_path: str,
    tmp_dir: str,
    scene_num: int,
) -> tuple[str, list[str]]:
    """
    Pre-render each 2-word chunk as a PNG, save to tmp_dir, and generate caps.txt
    for the FFmpeg concat demuxer.
    Returns (concat_txt_path, list_of_png_paths). Zero-lag, exact timing.
    """
    words = text.split()
    if not words:
        return "", []

    chunks = [
        " ".join(words[i: i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]
    chunk_dur = duration / max(len(chunks), 1)
    png_paths: list[str] = []

    for i, chunk in enumerate(chunks):
        t_start = i * chunk_dur
        t_end   = (i + 1) * chunk_dur

        clean_chunk, emoji_str = emoji_for_chunk(chunk)
        img = render_caption_png(clean_chunk, emoji_str, bold_font_path, emoji_font_path)
        png_path = os.path.join(tmp_dir, f"cap_s{scene_num}_{i:03d}.png")
        img.save(png_path, "PNG")
        png_paths.append(png_path)
        log.info(
            "Cap PNG s%d[%d] '%s%s' [%.3f,%.3f] size=%dx%d",
            scene_num, i, clean_chunk, emoji_str, t_start, t_end, img.width, img.height,
        )

    if not png_paths:
        return "", []

    concat_txt_path = os.path.join(tmp_dir, f"caps_s{scene_num}.txt")
    with open(concat_txt_path, "w", encoding="utf-8") as f:
        for p in png_paths:
            normalized_path = p.replace("\\", "/")
            f.write(f"file '{normalized_path}'\n")
            f.write(f"duration {chunk_dur:.4f}\n")
        # FFmpeg concat demuxer requirement: repeat last entry
        normalized_last = png_paths[-1].replace("\\", "/")
        f.write(f"file '{normalized_last}'\n")

    return concat_txt_path, png_paths


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
    log.info("Render request received (v4 Pillow captions + Concat Demuxer + High Quality)")
    scene_data = json.loads(scenes)
    uploads    = {1: video1, 2: video2, 3: video3, 4: video4}
    bold_font  = _find_font(_BOLD_FONT_CANDIDATES)
    emoji_font = _find_font(_EMOJI_FONT_CANDIDATES)
    log.info("Caption bold font: %s", bold_font or "PIL default")
    log.info("Caption emoji font: %s", emoji_font or "PIL default")

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

            # ── Pre-render caption PNGs & concat.txt via Pillow ─────────────
            concat_txt_path, png_paths = make_caption_overlays(
                scene["text"], duration, bold_font, emoji_font, tmp, num
            )

            # ── Build FFmpeg command ────────────────────────────────────────
            # If captions exist, feed video (#0) + caps concat stream (#1)
            # Uses 1 single overlay filter for zero lag and minimal RAM
            if concat_txt_path:
                ok, stderr = run_ffmpeg([
                    "ffmpeg", "-y",
                    "-stream_loop", "-1", "-i", raw_path,
                    "-f", "concat", "-safe", "0", "-i", concat_txt_path,
                    "-t", str(duration),
                    "-filter_complex", (
                        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[base];"
                        f"[base][1:v]overlay=x=(W-w)/2:y=H-h-{CAPTION_BOTTOM_PAD}[vout]"
                    ),
                    "-map", "[vout]",
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
            else:
                ok, stderr = run_ffmpeg([
                    "ffmpeg", "-y",
                    "-stream_loop", "-1", "-i", raw_path,
                    "-t", str(duration),
                    "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
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

            # Clean up raw + caption files
            os.remove(raw_path)
            if concat_txt_path and os.path.exists(concat_txt_path):
                os.remove(concat_txt_path)
            for png_path in png_paths:
                try:
                    os.remove(png_path)
                except OSError:
                    pass

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
