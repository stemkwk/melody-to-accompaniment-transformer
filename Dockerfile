# CPU-only INFERENCE image — reproducible accompaniment generation + WAV render
# on any machine with Docker. No GPU / CUDA required (37.9M model runs on CPU).
#
# Build:  docker build -t jam-infer .
# Run  :  docker run --rm -p 7860:7860 -v "$PWD/checkpoints:/app/checkpoints" jam-infer
#         → open http://localhost:7860  (Gradio demo)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

# System libs: fluidsynth + a General-MIDI soundfont (auto-detected by the code)
# for MIDI→WAV; libsndfile for soundfile/pedalboard.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fluidsynth libfluidsynth3 fluid-soundfont-gm libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heavy layer first (CPU PyTorch), then pinned inference deps — cached unless
# requirements change.
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-inference.txt .
RUN pip install -r requirements-inference.txt

# Package source (inference only — training/wandb not installed).
COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY app.py .
RUN pip install -e . --no-deps

EXPOSE 7860
# Checkpoints are mounted at runtime (not baked — they live in a GitHub Release).
CMD ["python", "app.py", \
     "--checkpoint", "checkpoints/best-epoch=007-val_loss=0.8431.ckpt", \
     "--host", "0.0.0.0", "--port", "7860"]
