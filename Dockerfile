# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps:
# - ffmpeg: required by Whisper/faster-whisper audio decoding in many setups
# - libsndfile1: common dependency for soundfile
# - build-essential: some wheels may compile on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency metadata first for better caching
# (works if you have pyproject.toml / setup.cfg / setup.py)
COPY pyproject.toml* setup.cfg* setup.py* README.md* requirements*.txt* ./

# Install project deps (editable install so CLI + src layout work)
# If you have extras like .[dev] or .[web], you can switch to those.
RUN pip install --upgrade pip && \
    pip install -e .

# Now copy the actual code
COPY . .

# Streamlit defaults
EXPOSE 8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

# Put runs in a volume-friendly location
RUN mkdir -p /app/runs

# Start the Streamlit app
CMD ["streamlit", "run", "src/asrle/dashboard/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
