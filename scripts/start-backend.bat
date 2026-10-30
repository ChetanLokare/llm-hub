@echo off
REM Starts the LLM Hub backend (Windows Command Prompt). Run from anywhere.
REM Optional - local Ollama: remove "REM " from the next four lines.
REM set LLMHUB_COMPATIBLE_BASE_URL=http://localhost:11434/v1
REM set LLMHUB_COMPATIBLE_LABEL=Ollama (local)
REM set LLMHUB_COMPATIBLE_API_KEY=ollama
REM set LLMHUB_COMPATIBLE_MODELS=qwen3:4b,gemma3:4b
cd /d "%~dp0..\backend"
if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) else if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat)
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
