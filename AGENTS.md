# etzhayyim-project-runpod — RunPod Serverless (Ollama)

## Overview

This project runs **Ollama on RunPod Serverless** behind a Cloudflare Worker gateway.

Current architecture:

- CF Worker: `runpod.etzhayyim.com`
- RunPod Serverless endpoint: `8rf4i80jpud0w2` (template: `1l08sdg6cc`)
- Inference engine: `ollama serve` (llama.cpp CUDA)
- Model: `gemma4:26b-a4b-it-q4_K_M` (MoE 26B total, 4B active, ~15.2GB Q4_K_M)

## Current Runtime Behavior (2026-04-11)

- Model is **pulled on worker startup** (`/api/pull`) and then warmed up (`/api/generate`)
- `OLLAMA_NUM_PARALLEL` and `CONCURRENCY` are set to `auto`
- Parallel slots derived from VRAM (gemma4:26b-a4b 24GB budget):
  - 24GB → ~5 slots (15.2GB weights + 7.8GB KV cache)
  - fallback → 2

## Worker Scaling Policy

Current endpoint settings:

- `workersMin=0` (scale to zero when idle)
- `workersMax=1` (single worker — cost control)
- `idleTimeout=60`
- `scalerType=QUEUE_DELAY`
- `scalerValue=1`
- GPU pools: `ADA_24, AMPERE_24, ADA_48_PRO`

Meaning:

- No requests: worker scales to 0
- Requests queue: 1 worker spins up (cold start ~90s for model pull)

## GPU Priority Order

Use the following GPU priority order to avoid low-supply primary GPU issues:

1. `NVIDIA L4`
2. `NVIDIA GeForce RTX 3090`
3. `NVIDIA GeForce RTX 4090`
4. `NVIDIA RTX A4000`
5. `NVIDIA RTX A4500`
6. `NVIDIA RTX 4000 Ada Generation`
7. `NVIDIA RTX 2000 Ada Generation`

## Known Failure and Fix

### Symptom

`404 Not Found` on `http://localhost:11434/api/generate` during warmup.

### Cause

Worker starts before model is locally available, then warmup call hits missing model path.

### Fix in `serve/handler.py`

- Always call `POST /api/pull` before warmup
- If first warmup returns 404, retry one `pull + generate`
- Raise explicit runtime error including Ollama response detail on failure

## Image and Template Strategy

### Why digest pinning

Using `:latest` can leave workers on stale cached images.
Use digest-pinned templates for deterministic rollouts.

Example image reference:

`ghcr.io/etzhayyim/runpod-ollama-gemma4@sha256:<digest>`

## Key Files

- `serve/handler.py` — RunPod handler + Ollama lifecycle + OpenAI-compatible proxying
- `serve/Dockerfile` — CUDA runtime + Ollama + Python SDK
- `serve/worker-gateway.ts` — Cloudflare Worker gateway to RunPod endpoint
- `serve/setup-endpoint.sh` — setup guide / helper

## Deployment Checklist

1. Build and push image to GHCR
2. Create or update template with digest-pinned image
3. Create endpoint with GPU priority order above
4. Set `workersMin=0`, `workersMax=2`
5. Verify:
   - `GET /v2/{endpointId}/health` -> worker ready after first request
   - `POST /v2/{endpointId}/runsync` -> `COMPLETED`
6. Deploy/update Cloudflare Worker secrets if endpoint ID changed

## API Keys

Required secrets (do not hardcode in repo files):

- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`
- gateway auth key(s) used by CF Worker
