# ── Shorts-Auto render service ──────────────────────────────────────────────
# Provides:
#   POST /tts    – Edge-TTS voiceover generation (MP3)
#   POST /render – FFmpeg video stitching (MP4)
#
# Build:  docker build -t shorts-auto .
# Run:    docker run -p 8000:8000 shorts-auto
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Install FFmpeg and the DejaVu Bold font used by the drawtext filter
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
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
