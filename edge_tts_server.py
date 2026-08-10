"""
Free Edge-TTS + Render HTTP wrapper for n8n - v6 (Pillow captions, 2-pass, 4K-safe)
-------------------------------------------------------------------------------------
Provides two endpoints:
  POST /tts    – generate MP3 voiceover via Microsoft Edge TTS (free, no key)
  POST /render – multi-scene render: 4 video clips + 1 voiceover + scenes JSON
                 → single vertical Short with:
                   • Pillow-rendered caption PNGs (true color emoji, exact timing)
                   • Bold yellow text, thick black stroke, drop shadow
                   • Auto emoji injection (keyword → 🔥💰⭐🏆⚡🧠😂😱🔑❤️)
                   • 2-word chunks, accurate timing via FFmpeg concat demuxer
                   • Captions lower-third (430 px above bottom)
                   • Cinema vignette overlay per scene
                   • Two-pass per scene: 4K → 1080p (Pass 1), then caption overlay (Pass 2)
                     This keeps RAM under 400 MB on AWS free tier even with 4K input.

SETUP:
  Docker:
    docker build -t shorts-auto . && docker run -p 8000:8000 shorts-auto

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
# Pass 1 (4K → 1080p prescale): ultrafast + CRF 30 = tiny intermediate file
# Pass 2 (caption overlay on 1080p): ultrafast + CRF 23 = decent final quality
PRESET_SCALE   = "ultrafast"
CRF_SCALE      = "30"
PRESET_CAPTION = "ultrafast"
CRF_CAPTION    = "23"
THREADS        = "1"
MAX_BITRATE    = "4000k"
BUF_SIZE       = "8000k"
AUDIO_BR       = "96k"
MAX_INPUT_MB   = 80

# ── Caption layout ────────────────────────────────────────────────────────────
WORDS_PER_CHUNK  = 2     # words shown at once
CAPTION_FONT_SIZE = 82   # Pillow font size (px)
CAPTION_STROKE_W  = 5    # black border thickness
CAPTION_SHADOW_OFF = 5   # drop shadow offset
CAPTION_BOTTOM_PAD = 430 # px from bottom of 1920px frame
# ALL PNGs MUST be this exact size — the concat demuxer feeds one overlay filter;
# any dimension change forces FFmpeg to reconfigure the filter graph → crash.
CAPTION_PNG_W  = 1060    # fixed canvas width
CAPTION_PNG_H  = 160     # fixed canvas height (fits 82px font + stroke + shadow)

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

app = FastAPI(title="Shorts Auto", version="6.0.0")


# ── Font helpers ───────────────────────────────────────────────────────────────

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


# ── Emoji lookup ───────────────────────────────────────────────────────────────

def emoji_for_chunk(chunk: str) -> str:
    """Return an emoji string to append to chunk, or '' if no match."""
    for word in chunk.lower().split():
        clean = word.strip(".,!?;:")
        if clean in EMOJI_MAP:
            return " " + EMOJI_MAP[clean]
    return ""


# ── Pillow caption rendering ──────────────────────────────────────────────────

def render_caption_png(
    text: str,
    bold_font_path: str,
    emoji_font_path: str,
) -> Image.Image:
    """
    Render a caption as a fixed-size (CAPTION_PNG_W × CAPTION_PNG_H) transparent
    RGBA PNG using Pillow.

    CRITICAL: All PNGs must be EXACTLY CAPTION_PNG_W × CAPTION_PNG_H.
    The FFmpeg concat demuxer feeds them into a single overlay filter; any size
    change triggers "Reconfiguring filter graph" → 0 output frames.

    Text + emoji are horizontally centered, vertically centered in the canvas.
    """
    bold_font  = _load_font(bold_font_path, CAPTION_FONT_SIZE)
    emoji_font = _load_font(emoji_font_path, CAPTION_FONT_SIZE) if emoji_font_path else None

    # Append emoji based on keywords
    emoji_str = emoji_for_chunk(text)
    label = text + emoji_str      # full caption string (text + optional emoji)

    # ── Fixed canvas ──────────────────────────────────────────────────────────
    img  = Image.new("RGBA", (CAPTION_PNG_W, CAPTION_PNG_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Measure on probe canvas to avoid clipping
    probe      = Image.new("RGBA", (CAPTION_PNG_W * 2, CAPTION_PNG_H * 4), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)

    # Measure text portion
    bbox_text = probe_draw.textbbox(
        (0, 0), text, font=bold_font, stroke_width=CAPTION_STROKE_W
    )
    text_w = bbox_text[2] - bbox_text[0]
    text_h = bbox_text[3] - bbox_text[1]

    # Measure emoji portion
    emoji_w = 0
    ef = emoji_font if emoji_font else bold_font
    if emoji_str:
        bbox_em = probe_draw.textbbox((0, 0), emoji_str.strip(), font=ef)
        emoji_w = bbox_em[2] - bbox_em[0] + 10  # 10px gap

    total_w = text_w + emoji_w

    # Center text block horizontally, center vertically
    x = (CAPTION_PNG_W - total_w) // 2 - bbox_text[0]
    y = (CAPTION_PNG_H - text_h) // 2 - bbox_text[1]

    # 1. Drop shadow
    draw.text(
        (x + CAPTION_SHADOW_OFF, y + CAPTION_SHADOW_OFF),
        text, font=bold_font, fill=(0, 0, 0, 180),
    )

    # 2. Main text — yellow with black stroke
    draw.text(
        (x, y), text, font=bold_font,
        fill=(255, 230, 0, 255),
        stroke_width=CAPTION_STROKE_W,
        stroke_fill=(0, 0, 0, 255),
    )

    # 3. Emoji — right of text
    if emoji_str:
        x_emoji = x + text_w + bbox_text[0] + 10
        try:
            if emoji_font:
                draw.text((x_emoji, y), emoji_str.strip(), font=ef, embedded_color=True)
            else:
                draw.text(
                    (x_emoji, y), emoji_str.strip(), font=ef,
                    fill=(255, 230, 0, 255),
                    stroke_width=CAPTION_STROKE_W,
                    stroke_fill=(0, 0, 0, 255),
                )
        except Exception:
            draw.text(
                (x_emoji, y), emoji_str.strip(), font=bold_font,
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
) -> str:
    """
    Pre-render each 2-word chunk as a Pillow PNG, save to tmp_dir,
    and write a concat.txt for the FFmpeg concat demuxer.
    Returns the path to caps_sN.txt (or '' if text is empty).
    """
    words = text.split()
    if not words:
        return ""

    chunks = [
        " ".join(words[i: i + WORDS_PER_CHUNK])
        for i in range(0, len(words), WORDS_PER_CHUNK)
    ]
    chunk_dur = duration / max(len(chunks), 1)
    png_paths: list[str] = []

    for i, chunk in enumerate(chunks):
        img = render_caption_png(chunk, bold_font_path, emoji_font_path)
        assert img.size == (CAPTION_PNG_W, CAPTION_PNG_H), \
            f"PNG size mismatch: {img.size} != {(CAPTION_PNG_W, CAPTION_PNG_H)}"
        png_path = os.path.join(tmp_dir, f"cap_s{scene_num}_{i:03d}.png")
        img.save(png_path, "PNG")
        png_paths.append(png_path)
        log.info(
            "Cap PNG s%d[%d/%d] %r → [%.3f, %.3f]",
            scene_num, i, len(chunks) - 1, chunk, i * chunk_dur, (i + 1) * chunk_dur,
        )

    concat_path = os.path.join(tmp_dir, f"caps_s{scene_num}.txt")
    with open(concat_path, "w", encoding="utf-8") as f:
        for p in png_paths:
            f.write(f"file '{p.replace(chr(92), '/')}'\n")
            f.write(f"duration {chunk_dur:.6f}\n")
        # Sentinel: repeat last entry so demuxer knows the final frame PTS
        f.write(f"file '{png_paths[-1].replace(chr(92), '/')}'\n")

    return concat_path


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: list[str], label: str) -> tuple[bool, str]:
    """Run FFmpeg, log stderr. Returns (success, stderr)."""
    log.info("FFmpeg [%s]: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log.info("FFmpeg [%s] exit=%d\n%s", label, result.returncode, result.stderr)
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
    log.info("Render request received (v6 Pillow captions, 2-pass, 4K-safe)")
    scene_data = json.loads(scenes)
    uploads    = {1: video1, 2: video2, 3: video3, 4: video4}
    bold_font  = _find_font(_BOLD_FONT_CANDIDATES)
    emoji_font = _find_font(_EMOJI_FONT_CANDIDATES)
    log.info("Fonts: bold=%s  emoji=%s", bold_font or "(default)", emoji_font or "(none)")

    with tempfile.TemporaryDirectory() as tmp:

        # ── 1. Write audio ─────────────────────────────────────────────────
        audio_path  = os.path.join(tmp, "audio.mp3")
        audio_bytes = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        log.info("Audio: %d bytes", len(audio_bytes))
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

            scaled_path = os.path.join(tmp, f"scaled{num}.mp4")
            clip_path   = os.path.join(tmp, f"clip{num}.mp4")

            # ── Pass 1: 4K → 1080×1920 + vignette ────────────────────────
            # Single stream: decode 4K → scale → vignette → encode 1080p.
            # No second stream = no overlay = safe on 512 MB free-tier.
            ok, stderr = run_ffmpeg([
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-t", str(duration), "-i", raw_path,
                "-vf", (
                    "scale=1080:1920:force_original_aspect_ratio=increase,"
                    "crop=1080:1920,"
                    "vignette=angle=PI/4:mode=backward"
                ),
                "-an",
                "-c:v", "libx264",
                "-preset", PRESET_SCALE,
                "-crf", CRF_SCALE,
                "-threads", THREADS,
                "-r", "30",
                "-pix_fmt", "yuv420p",
                scaled_path,
            ], label=f"scale{num}")

            os.remove(raw_path)   # free disk immediately

            if not ok:
                return Response(
                    content=f"FFmpeg scale failed on scene {num}:\n{stderr[-3000:]}",
                    media_type="text/plain", status_code=500,
                )

            # ── Pre-render caption PNGs (Pillow) ──────────────────────────
            caps_txt = make_caption_overlays(
                scene["text"], duration, bold_font, emoji_font, tmp, num
            )

            # ── Pass 2: overlay Pillow captions onto 1080p clip ───────────
            # Input is already 1080p → trivial RAM. concat demuxer feeds
            # fixed-size PNGs into one overlay filter (eof_action=pass so
            # video continues if PNG stream ends slightly early).
            if caps_txt:
                ok, stderr = run_ffmpeg([
                    "ffmpeg", "-y",
                    "-i", scaled_path,
                    "-f", "concat", "-safe", "0", "-i", caps_txt,
                    "-filter_complex", (
                        f"[0:v][1:v]overlay="
                        f"x=(W-w)/2:"
                        f"y=H-h-{CAPTION_BOTTOM_PAD}:"
                        f"eof_action=pass[vout]"
                    ),
                    "-map", "[vout]",
                    "-an",
                    "-c:v", "libx264",
                    "-preset", PRESET_CAPTION,
                    "-crf", CRF_CAPTION,
                    "-maxrate", MAX_BITRATE,
                    "-bufsize", BUF_SIZE,
                    "-threads", THREADS,
                    "-r", "30",
                    "-pix_fmt", "yuv420p",
                    clip_path,
                ], label=f"caption{num}")
            else:
                # No text → just copy scaled clip as-is
                import shutil
                shutil.copy2(scaled_path, clip_path)
                ok = True

            # Cleanup
            os.remove(scaled_path)
            if caps_txt and os.path.exists(caps_txt):
                os.remove(caps_txt)
            # Remove PNGs
            for fname in os.listdir(tmp):
                if fname.startswith(f"cap_s{num}_") and fname.endswith(".png"):
                    try:
                        os.remove(os.path.join(tmp, fname))
                    except OSError:
                        pass

            if not ok:
                return Response(
                    content=f"FFmpeg caption failed on scene {num}:\n{stderr[-3000:]}",
                    media_type="text/plain", status_code=500,
                )

            processed_clips.append(clip_path)
            scene_durations.append(duration)
            log.info("Scene %d done → %s", num, clip_path)
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
            log.info("Concat done → %s", concatenated)

        # ── 4. Mux voiceover ──────────────────────────────────────────────
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

        log.info("Render complete (v6). Output: %.1f MB", len(video_bytes) / 1024 / 1024)
        return Response(content=video_bytes, media_type="video/mp4")


# ── Diagnostics ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    bold  = _find_font(_BOLD_FONT_CANDIDATES)
    emoji = _find_font(_EMOJI_FONT_CANDIDATES)
    return {
        "status":      "ok",
        "version":     "6.0.0",
        "bold_font":   bold  or "PIL-default",
        "emoji_font":  emoji or "none",
    }


@app.get("/logs")
async def get_logs():
    """Return the last 8 KB of the render log for remote diagnosis."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            return Response(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        return Response(content="Log not yet created.", media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
