# ── Shorts-Auto render service ──────────────────────────────────────────────
# Provides:
#   POST /tts    – Edge-TTS voiceover generation (MP3)
#   POST /render – FFmpeg video stitching (MP4)
#
# Build:  docker build -t shorts-auto .
# Run:    docker run -p 8000:8000 shorts-auto
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Install FFmpeg, DejaVu Bold (bold caption fallback), Liberation fonts
# (free Arial/Impact-like alternative), and Noto Color Emoji for emoji
# rendering in FFmpeg drawtext captions.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fonts-liberation \
        fonts-noto-color-emoji \
 && fc-cache -fv \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY edge_tts_server.py .

# Render.com / Railway expose $PORT; fall back to 8000 locally
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn edge_tts_server:app --host 0.0.0.0 --port ${PORT}"]
