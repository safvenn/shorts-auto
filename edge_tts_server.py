"""
Free Edge-TTS + Render HTTP wrapper for n8n - v4 (word-accurate captions)
---------------------------------------------------------------------------
Provides two endpoints:
  POST /tts    – generate MP3 voiceover via Microsoft Edge TTS (free, no key),
                 PLUS real per-word timing (from Edge TTS's own WordBoundary
                 events) returned in the `X-Word-Timings` response header.
  POST /render – multi-scene render: 4 video clips + 1 voiceover + scenes JSON
                 (+ optional wordTimings JSON) → single vertical Short with:
                   • Modern "karaoke" captions: white text, one short line of
                     up to 4 words on screen, the CURRENTLY-SPOKEN word lit
                     up in bright yellow — the same look used by CapCut,
                     Submagic, Opus Clip and most viral Shorts/Reels today.
                   • Captions centered mid-screen (clear of TikTok/IG/YouTube
                     UI chrome at the very top and bottom).
                   • Pixel-accurate word spacing, measured with the exact
                     caption font via Pillow (no more guessed widths).
                   • Word-accurate timing: if the caller forwards the
                     `wordTimings` produced by /tts, each word lights up at
                     the exact millisecond it's actually spoken. Falls back
                     to even spacing across the scene if not provided.
                   • Auto emoji injection (keyword → 🔥💰⭐🏆⚡🧠😂😱🔑❤️)
                   • Cinema vignette overlay per scene

SETUP:
  Docker (recommended for /render – FFmpeg required):
    docker build -t shorts-auto . && docker run -p 8000:8000 shorts-auto

  Local (TTS-only, no FFmpeg):
    pip install fastapi uvicorn edge-tts python-multipart pillow
    python edge_tts_server.py

USAGE:
  POST /tts
    Body (JSON): { "text": "...", "voice": "en-US-GuyNeural" }
    Returns: MP3 audio bytes
    Header:  X-Word-Timings – base64 JSON array of [word, startSeconds,
             endSeconds] triples, timed against the returned audio.
             Forward this whole header value straight through as the
             `wordTimings` form field on /render for word-accurate captions.

  POST /render  (multipart/form-data)
    Fields:
      audio        – binary  full voiceover track (matches the full script)
      video1..4    – binary  scene 1-4 footage
      scenes       – text    JSON: [{"sceneNumber":1,"text":"...","durationSeconds":7}, ...]
      wordTimings  – text    OPTIONAL. Base64 JSON array of [word,start,end]
                     (in seconds, relative to the full `audio` track) as
                     returned in /tts's `X-Word-Timings` header. Omit to fall
                     back to even per-word spacing within each scene.
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

import base64
import gc
import io
import json
import logging
import os
import re
import subprocess
import tempfile

import edge_tts
from fastapi import FastAPI, Form, Response, UploadFile
from PIL import ImageFont
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
CRF          = "28"
THREADS      = "1"
MAX_BITRATE  = "900k"
BUF_SIZE     = "1800k"
AUDIO_BR     = "96k"
MAX_INPUT_MB = 80

FRAME_W, FRAME_H = 1080, 1920

# ── Caption style constants (standard modern Shorts/TikTok "karaoke" look) ──
WORDS_PER_LINE     = 4          # soft cap on words per line, like CapCut/Submagic defaults
CAPTION_FONTSIZE   = 84         # big, legible on a phone screen
CAPTION_Y_RATIO    = 0.52       # vertical center-ish — clear of top/bottom app UI
CAPTION_STROKE_W   = 12         # thick black outline, standard for readability
WORD_GAP_PX        = 20         # gap between words on the same line
SIDE_MARGIN_PX     = 48         # minimum margin kept clear on each side
MAX_LINE_WIDTH_PX  = FRAME_W - 2 * SIDE_MARGIN_PX  # hard cap — lines never overflow the frame
BASE_WORD_COLOR    = "white"
ACTIVE_WORD_COLOR  = "0xFFE600" # bright yellow — the word being spoken right now

# Edge TTS MP3 encoder delay (seconds). WordBoundary offsets are measured from
# the TTS engine's internal clock, but the MP3 container has a small encoder
# priming delay (~0.05 s on most Edge voices). Subtracting this value shifts
# captions forward so they align with the actual audio playback. Tune to 0.0
# to disable. Values between 0.04 and 0.08 are typical.
EDGE_TTS_AUDIO_OFFSET_S = 0.05

# Cycle of xfade transition names (FFmpeg built-ins) — currently unused (see
# render_short: concat uses stream-copy for RAM safety) but kept for future use.
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

app = FastAPI(title="Shorts Auto", version="4.0.0")


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


def clean_word(word: str) -> str:
    """Strip stray leading/trailing commas, periods, and whitespace."""
    return re.sub(r"^[,.\s]+|[,.\s]+$", "", word)


def emoji_for_chunk(chunk: str) -> str:
    """Return a single emoji for a line of text based on keyword lookup, or ''.

    NOTE: Emoji are intentionally NOT injected into FFmpeg drawtext filters
    because FreeType (used by FFmpeg drawtext) cannot render color emoji glyphs
    — they appear as blank boxes. This function is kept for future use with an
    image-overlay approach (e.g. Pillow composite frames) but currently returns
    an empty string so no boxes appear in the rendered video.
    """
    return ""


def build_font_path() -> str:
    """
    Return the best available bold font path on the system.
    Priority: Montserrat ExtraBold / Poppins ExtraBold (authentic Shorts-caption
    look, installed at build time) → DejaVuSans-Bold → Liberation Bold → "".
    """
    candidates = [
        "/usr/share/fonts/truetype/shorts/Montserrat-ExtraBold.ttf",
        "/usr/share/fonts/truetype/shorts/Poppins-ExtraBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            log.info("Caption font: %s", path)
            return path
    log.warning("No preferred font found; FFmpeg will use its default.")
    return ""


_PIL_FONT_CACHE: dict[tuple[str, int], "ImageFont.FreeTypeFont"] = {}


def load_pil_font(font_path: str, size: int):
    """Load (and cache) a Pillow font object for precise text measurement."""
    key = (font_path, size)
    if key in _PIL_FONT_CACHE:
        return _PIL_FONT_CACHE[key]
    try:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not load font %s for measurement: %s", font_path, exc)
        font = ImageFont.load_default()
    _PIL_FONT_CACHE[key] = font
    return font


def measure_width(font, text: str) -> int:
    """Pixel width of `text` rendered in `font`, matching FFmpeg's own font metrics."""
    if not text:
        return 0
    bbox = font.getbbox(text)
    return max(bbox[2] - bbox[0], 1)


def decode_word_timings(raw):
    """
    Decode the `wordTimings` form field. Accepts either the raw base64 string
    produced by /tts's X-Word-Timings header, or a plain JSON array — so the
    endpoint is forgiving about exactly how the caller forwards it.
    """
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception as exc:
        log.warning("Could not decode wordTimings, falling back to even spacing: %s", exc)
        return None


def split_word_timings_by_scene(word_timings, scenes_sorted):
    """
    Distribute a flat, full-narration word-timing list across scenes, based on
    each scene's durationSeconds, and re-base each word's time to be local to
    its own scene's clip (0 = start of that scene's video).
    """
    result = {}
    cursor = 0.0
    last_num = scenes_sorted[-1]["sceneNumber"] if scenes_sorted else None

    for scene in scenes_sorted:
        num = scene["sceneNumber"]
        dur = float(scene["durationSeconds"])
        scene_start, scene_end = cursor, cursor + dur
        bucket = []

        for entry in word_timings:
            word, start, end = entry[0], float(entry[1]), float(entry[2])
            midpoint = (start + end) / 2
            in_range = scene_start <= midpoint < scene_end
            if not in_range and num == last_num and midpoint >= scene_end:
                in_range = True  # catch any trailing words on the final scene
            if not in_range:
                continue
            local_start = max(0.0, start - scene_start)
            local_end = min(dur, end - scene_start)
            if local_end > local_start:
                bucket.append({"word": word, "start": local_start, "end": local_end})

        result[num] = bucket
        cursor = scene_end

    return result


def build_caption_words(scene_text, duration, timed_words):
    """
    Return a flat list of {word, start, end} (seconds, local to this scene)
    for a scene — using real per-word timing when available, otherwise
    falling back to evenly spacing the scene's own words across its duration.
    """
    if timed_words:
        return [
            {"word": clean_word(w["word"]), "start": w["start"], "end": w["end"]}
            for w in timed_words
            if clean_word(w["word"])
        ]

    words = [clean_word(w) for w in scene_text.split()]
    words = [w for w in words if w]
    if not words:
        return []
    step = duration / len(words)
    return [
        {"word": w, "start": i * step, "end": (i + 1) * step}
        for i, w in enumerate(words)
    ]


def wrap_words_by_width(words, pil_font):
    """
    Group words into caption lines using each word's real measured pixel
    width, capped at both WORDS_PER_LINE and MAX_LINE_WIDTH_PX — so a line
    never overflows the frame even when it contains long words.
    Returns (lines, display_widths) where display_widths[i][j] is the
    pixel width of the j-th word (already uppercased) in lines[i].
    """
    lines: list[list[dict]] = []
    widths: list[list[int]] = []

    current: list[dict] = []
    current_w: list[int] = []
    current_total = 0

    for w in words:
        txt = w["word"].upper()
        w_px = measure_width(pil_font, txt)
        added_w = w_px if not current else w_px + WORD_GAP_PX

        would_overflow = current and (current_total + added_w > MAX_LINE_WIDTH_PX)
        would_exceed_count = len(current) >= WORDS_PER_LINE

        if current and (would_overflow or would_exceed_count):
            lines.append(current)
            widths.append(current_w)
            current, current_w, current_total = [], [], 0
            added_w = w_px

        current.append(w)
        current_w.append(w_px)
        current_total += added_w

    if current:
        lines.append(current)
        widths.append(current_w)

    return lines, widths


def caption_drawtext_filters(scene_text, duration, font_path, timed_words):
    """
    Build FFmpeg drawtext filters for modern "karaoke" captions:
    - Words wrapped into lines by real measured pixel width (never overflows
      the frame), pixel-accurately centered (measured with Pillow, using the
      same font file FFmpeg will render with)
    - Every word sits in place as white text for its whole line's on-screen
      window, then lights up bright yellow for the exact span it's spoken
    - Auto emoji appended to the last word of each line where relevant
    """
    words = build_caption_words(scene_text, duration, timed_words)
    if not words:
        return []

    pil_font = load_pil_font(font_path, CAPTION_FONTSIZE)
    font_arg = f":fontfile='{font_path}'" if font_path else ""

    lines, line_widths = wrap_words_by_width(words, pil_font)

    filters = []

    for line, widths in zip(lines, line_widths):
        line_start = line[0]["start"]
        line_end = min(line[-1]["end"], duration)
        line_text = " ".join(w["word"] for w in line)
        emoji_suffix = emoji_for_chunk(line_text)

        display = [w["word"].upper() for w in line]
        if emoji_suffix:
            display[-1] = display[-1] + emoji_suffix
            widths = widths[:-1] + [measure_width(pil_font, display[-1])]

        total_w = sum(widths) + WORD_GAP_PX * (len(display) - 1)
        cursor_x = max(SIDE_MARGIN_PX, (FRAME_W - total_w) // 2)

        enable_line = f"gte(t\\,{line_start:.4f})*lt(t\\,{line_end:.4f})"

        for word_info, txt, w_px in zip(line, display, widths):
            escaped = escape_drawtext(txt)
            x = cursor_x
            cursor_x += w_px + WORD_GAP_PX

            shared_style = (
                f":fontsize={CAPTION_FONTSIZE}"
                f":bordercolor=black:borderw={CAPTION_STROKE_W}"
                f":shadowcolor=black@0.6:shadowx=3:shadowy=3"
                f":x={x}:y=h*{CAPTION_Y_RATIO}"
                f"{font_arg}"
            )

            # Base layer: plain white, visible for the whole line's window
            filters.append(
                f"drawtext=text='{escaped}'"
                f":fontcolor={BASE_WORD_COLOR}"
                f":enable={enable_line}"
                + shared_style
            )

            # Highlight layer: bright yellow, only while THIS word is spoken —
            # drawn on top of the base layer at the identical position, so it
            # reads as the word "lighting up" exactly on beat with the audio.
            word_start = max(word_info["start"], line_start)
            word_end = min(word_info["end"], line_end)
            if word_end > word_start:
                enable_word = f"gte(t\\,{word_start:.4f})*lt(t\\,{word_end:.4f})"
                filters.append(
                    f"drawtext=text='{escaped}'"
                    f":fontcolor={ACTIVE_WORD_COLOR}"
                    f":enable={enable_word}"
                    + shared_style
                )

    return filters


def vignette_filter() -> str:
    """Cinema edge-darkening vignette."""
    return "vignette=angle=PI/4:mode=backward"


def run_ffmpeg(cmd, label: str):
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
    word_timings = []

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            # offset/duration are in 100-nanosecond units.
            # We subtract EDGE_TTS_AUDIO_OFFSET_S to compensate for the MP3
            # encoder priming delay — without this the captions run ~50 ms late
            # relative to the actual audio playback.
            raw_start = chunk["offset"] / 1e7
            raw_end   = (chunk["offset"] + chunk["duration"]) / 1e7
            start = round(max(0.0, raw_start - EDGE_TTS_AUDIO_OFFSET_S), 3)
            end   = round(max(start + 0.001, raw_end - EDGE_TTS_AUDIO_OFFSET_S), 3)
            word_timings.append([chunk["text"], start, end])

    audio_buffer.seek(0)
    audio_bytes = audio_buffer.read()

    timings_b64 = base64.b64encode(
        json.dumps(word_timings, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")

    log.info("TTS generated: %d bytes audio, %d words timed", len(audio_bytes), len(word_timings))

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"X-Word-Timings": timings_b64},
    )


# ── Render ────────────────────────────────────────────────────────────────────

@app.post("/render")
async def render_short(
    audio: UploadFile,
    video1: UploadFile,
    video2: UploadFile,
    video3: UploadFile,
    video4: UploadFile,
    scenes: str = Form(...),
    wordTimings: str = Form(None),
):
    log.info("Render request received (v4 word-accurate captions)")
    scene_data = sorted(json.loads(scenes), key=lambda s: s["sceneNumber"])
    uploads = {1: video1, 2: video2, 3: video3, 4: video4}
    font_path = build_font_path()

    raw_word_timings = decode_word_timings(wordTimings)
    per_scene_words = (
        split_word_timings_by_scene(raw_word_timings, scene_data)
        if raw_word_timings
        else {}
    )
    log.info(
        "Word timings: %s",
        "using real per-word timing" if raw_word_timings else "none supplied, using even spacing",
    )

    with tempfile.TemporaryDirectory() as tmp:

        # ── 1. Write audio ────────────────────────────────────────────────
        audio_path = os.path.join(tmp, "audio.mp3")
        audio_bytes = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        log.info("Audio written: %d bytes", len(audio_bytes))
        del audio_bytes
        gc.collect()

        processed_clips = []

        # ── 2. Process each scene independently ───────────────────────────
        for scene in scene_data:
            num = scene["sceneNumber"]
            duration = float(scene["durationSeconds"])
            log.info("Scene %d | duration=%.1fs | text=%r", num, duration, scene["text"])

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
            del raw_bytes
            gc.collect()

            clip_path = os.path.join(tmp, f"clip{num}.mp4")

            scale_crop = (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920"
            )
            vignette = vignette_filter()
            captions = caption_drawtext_filters(
                scene["text"], duration, font_path, per_scene_words.get(num)
            )

            vf = ",".join([scale_crop, vignette] + captions)

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
            log.info("Scene %d encoded → %s", num, clip_path)
            gc.collect()

        # ── 3. Concat clips (stream-copy, zero RAM) ────────────────────────
        n = len(processed_clips)

        if n == 1:
            concatenated = processed_clips[0]
            log.info("Single scene – skipping concat")
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
                    content=f"FFmpeg concat failed. Last stderr:\n{stderr[-3000:]}",
                    media_type="text/plain",
                    status_code=500,
                )
            log.info("Concat done → %s", concatenated)

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

        log.info("Render complete (v4). Output size: %.1f MB", len(video_bytes) / (1024 * 1024))
        return Response(content=video_bytes, media_type="video/mp4")


# ── Diagnostics ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    font = build_font_path()
    return {
        "status": "ok",
        "service": "ffmpeg-render-wrapper-v4-word-accurate-captions",
        "version": "4.0.0",
        "caption_font": font or "ffmpeg-default",
        "caption_style": {
            "words_per_line": WORDS_PER_LINE,
            "fontsize": CAPTION_FONTSIZE,
            "y_ratio": CAPTION_Y_RATIO,
            "base_color": BASE_WORD_COLOR,
            "active_word_color": ACTIVE_WORD_COLOR,
        },
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