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
# rendering in FFmpeg drawtext captions. Also fetch Montserrat/Poppins
# ExtraBold — free (OFL-licensed) fonts that match the look most modern
# YouTube Shorts / TikTok caption tools (CapCut, Submagic, Opus Clip) use.
# The `|| true` on each download means a network hiccup or a moved file at
# build time can't break the build — build_font_path() in the app falls
# back through DejaVu/Liberation automatically if these aren't present.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fonts-liberation \
        fonts-noto-color-emoji \
        curl \
 && mkdir -p /usr/share/fonts/truetype/shorts \
 && curl -fsSL -o /usr/share/fonts/truetype/shorts/Montserrat-ExtraBold.ttf \
        https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-ExtraBold.ttf || true \
 && curl -fsSL -o /usr/share/fonts/truetype/shorts/Poppins-ExtraBold.ttf \
        https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-ExtraBold.ttf || true \
 && find /usr/share/fonts/truetype/shorts -size -10k -delete \
 && apt-get purge -y --auto-remove curl \
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