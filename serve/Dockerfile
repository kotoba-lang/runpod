# RunPod Ollama — lightweight image (model pulled on first start)
#
# Image size: ~2GB (no model baked in)
# Model download on cold start: ~15.2GB gemma4:26b-a4b-it-q4_K_M (~90s on RunPod network)
#
# Build:
#   docker build --platform linux/amd64 -t ghcr.io/etzhayyim/runpod-ollama-gemma4:v3 .

FROM runpod/worker-vllm:stable-cuda12.1.0 AS base

# We only need the CUDA runtime + Python from this image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl python3 python3-pip ca-certificates zstd pciutils \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install RunPod SDK + HTTP client
RUN pip3 install --no-cache-dir \
    runpod>=1.7.0 httpx>=0.27.0

WORKDIR /app
COPY handler.py /app/handler.py

# Ollama config (model NOT baked in — pulled on first start)
ENV OLLAMA_MODEL="gemma4:26b-a4b-it-q4_K_M"
ENV OLLAMA_HOST="http://localhost:11434"
ENV OLLAMA_NUM_PARALLEL="auto"
ENV OLLAMA_MAX_LOADED_MODELS="1"
ENV OLLAMA_FLASH_ATTENTION="1"
ENV OLLAMA_KV_CACHE_TYPE="q8_0"
ENV CONCURRENCY="auto"

CMD ["python3", "-u", "/app/handler.py"]
