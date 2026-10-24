#!/usr/bin/env bash
# Build llama.cpp (if needed) and run a short text prompt on Gemma 4 E2B and E4B.
# With --audio, also run one native multimodal turn on E2B using the mmproj file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/third_party/llama.cpp"
BUILD="$LLAMA/build"
BIN="$BUILD/bin/llama-cli"
LLM="$ROOT/models/llm"
MM="$ROOT/models/mmproj"
PROMPT="${PROMPT:-用一句中文介绍你自己。}"
N_PREDICT="${N_PREDICT:-48}"
CTX="${CTX:-2048}"

if [[ ! -x "$BIN" ]]; then
  echo "==> building llama-cli"
  cmake -S "$LLAMA" -B "$BUILD" -DGGML_METAL=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON
  cmake --build "$BUILD" --target llama-cli -j"$(sysctl -n hw.ncpu)"
fi

run_text() {
  local name="$1"
  local model="$2"
  echo
  echo "======== TEXT $name ========"
  "$BIN" -m "$model" -c "$CTX" -n "$N_PREDICT" --temp 1.0 --top-p 0.95 --top-k 64 \
    -p "$PROMPT" --no-display-prompt
}

run_text "E2B" "$LLM/gemma-4-E2B-it-Q4_K_M.gguf"
run_text "E4B" "$LLM/gemma-4-E4B-it-Q4_K_M.gguf"

if [[ "${1:-}" == "--audio" ]]; then
  WAV="${WAV:-$ROOT/models/samples/hello.wav}"
  if [[ ! -f "$WAV" ]]; then
    echo "missing wav: $WAV" >&2
    exit 1
  fi
  echo
  echo "======== NATIVE AUDIO E2B ========"
  "$BIN" -m "$LLM/gemma-4-E2B-it-Q4_K_M.gguf" \
    --mmproj "$MM/mmproj-gemma-4-E2B-it-Q8_0.gguf" \
    --audio "$WAV" -c "$CTX" -n "$N_PREDICT" --temp 1.0 \
    -p "Transcribe the audio, then answer in one short sentence." \
    --no-display-prompt
fi

echo
echo "local gemma4 test finished"
