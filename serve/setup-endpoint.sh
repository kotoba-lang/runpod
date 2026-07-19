#!/usr/bin/env bash
# RunPod Serverless endpoint setup — Ollama (gemma3:4b)
#
# Prerequisites:
#   1. RunPod account + API key
#   2. Docker logged in to GHCR
#
# Usage:
#   export RUNPOD_API_KEY="rpa_..."
#   bash setup-endpoint.sh
#
# Note:
#   This script builds/pushes image and prints recommended RunPod settings.

set -euo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
IMAGE="ghcr.io/etzhayyim/runpod-ollama-gemma4:latest"
CACHE_REF="${BUILDKIT_CACHE_REF:-ghcr.io/etzhayyim/build-cache:runpod-ollama-gemma4}"

echo "=== Step 1: Build & Push Docker Image ==="
cd "$(dirname "$0")"
docker buildx build --builder "${BUILDKIT_BUILDER:-etzhayyim-vke}" --platform linux/amd64 \
  --cache-from "type=registry,ref=${CACHE_REF}" \
  --cache-to "type=registry,ref=${CACHE_REF},mode=max" \
  --push -t "$IMAGE" .

echo ""
echo "=== Step 2: Create/Update RunPod Serverless Endpoint ==="
echo ""
echo "RunPod Console:"
echo "  https://www.runpod.io/console/serverless"
echo ""
echo "Recommended configuration:"
echo "  Docker Image:   $IMAGE"
echo "  Workers Min:    0"
echo "  Workers Max:    2"
echo "  Idle Timeout:   60 seconds"
echo "  Scaler Type:    QUEUE_DELAY"
echo "  Scaler Value:   1"
echo ""
echo "GPU priority order (in this exact order):"
echo "  1) NVIDIA L4"
echo "  2) NVIDIA GeForce RTX 3090"
echo "  3) NVIDIA GeForce RTX 4090"
echo "  4) NVIDIA RTX A4000"
echo "  5) NVIDIA RTX A4500"
echo "  6) NVIDIA RTX 4000 Ada Generation"
echo "  7) NVIDIA RTX 2000 Ada Generation"
echo ""
echo "Template env variables:"
echo "  OLLAMA_MODEL=gemma3:4b"
echo "  OLLAMA_HOST=http://localhost:11434"
echo "  OLLAMA_NUM_PARALLEL=auto"
echo "  OLLAMA_MAX_LOADED_MODELS=1"
echo "  OLLAMA_FLASH_ATTENTION=1"
echo "  OLLAMA_KV_CACHE_TYPE=q8_0"
echo "  CONCURRENCY=auto"
echo ""
echo "=== Step 3: Set Worker Secrets ==="
echo ""
echo "After endpoint creation, set CF Worker secrets:"
echo "  cd serve"
echo "  echo 'YOUR_ENDPOINT_ID' | wrangler secret put RUNPOD_ENDPOINT_ID"
echo "  echo '$RUNPOD_API_KEY' | wrangler secret put RUNPOD_API_KEY"
echo ""
echo "=== Step 4: Verify ==="
echo ""
echo "  curl https://runpod.etzhayyim.com/health"
echo "  curl -H 'x-api-key: rpgw_...' https://runpod.etzhayyim.com/v1/models"
echo ""
echo 'Direct RunPod test:'
echo '  curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \'
echo '    -H "Authorization: Bearer $RUNPOD_API_KEY" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"input":{"messages":[{"role":"user","content":"3+3=?"}],"max_tokens":100}}'"'"''
