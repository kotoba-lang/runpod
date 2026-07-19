#!/usr/bin/env bash
# Convert Gemma 4 E2B IT to GGUF and import into Ollama.
#
# Use this if the model is not yet in the Ollama library.
# Requires: python3, pip, ollama
#
# Usage:
#   export HF_TOKEN="hf_..."
#   bash create-model.sh

set -euo pipefail

MODEL_HF="google/gemma-4-e2b-it"
MODEL_OLLAMA="gemma4-e2b:q4_K_M"
WORK_DIR="/tmp/gemma4-e2b-gguf"

echo "=== Step 1: Install llama-cpp-python (for GGUF conversion) ==="
pip3 install --no-cache-dir huggingface-hub llama-cpp-python

echo "=== Step 2: Download HF model ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_HF}', local_dir='${WORK_DIR}/hf')
"

echo "=== Step 3: Convert to GGUF Q4_K_M ==="
# Use llama.cpp convert script
python3 -m llama_cpp.convert \
  --outfile "${WORK_DIR}/gemma4-e2b-it-q4_K_M.gguf" \
  --outtype q4_K_M \
  "${WORK_DIR}/hf"

echo "=== Step 4: Create Ollama Modelfile ==="
cat > "${WORK_DIR}/Modelfile" << 'EOF'
FROM ./gemma4-e2b-it-q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 8192

TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<start_of_turn>user
{{ .Content }}<end_of_turn>
<start_of_turn>model
{{ else if eq .Role "assistant" }}{{ .Content }}<end_of_turn>
{{ end }}{{ end }}"""
EOF

echo "=== Step 5: Import into Ollama ==="
cd "${WORK_DIR}"
ollama create "${MODEL_OLLAMA}" -f Modelfile

echo "=== Done ==="
echo "Model available as: ${MODEL_OLLAMA}"
echo "Test: ollama run ${MODEL_OLLAMA} 'Hello'"
