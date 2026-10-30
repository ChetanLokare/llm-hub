#!/usr/bin/env bash
# Starts the LLM Hub backend (macOS / Linux). Run from anywhere.
# Optional - local Ollama: uncomment these four lines.
# export LLMHUB_COMPATIBLE_BASE_URL=http://localhost:11434/v1
# export LLMHUB_COMPATIBLE_LABEL="Ollama (local)"
# export LLMHUB_COMPATIBLE_API_KEY=ollama
# export LLMHUB_COMPATIBLE_MODELS=qwen3:4b,gemma3:4b
set -euo pipefail
cd "$(dirname "$0")/../backend"
if [ -f venv/bin/activate ]; then . venv/bin/activate; elif [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi
exec uvicorn main:app --host 127.0.0.1 --port 8000 --reload
