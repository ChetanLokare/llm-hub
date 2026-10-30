# LLM Hub

One local chat window for many AI providers. Ask in text, attach images, video, audio, PDFs, documents, code or zip files,
and get text or generated images back. LLM Hub can pick the model for you, switch automatically when one fails, and keeps your
conversations, files and notes **on your own machine** so a change of model never loses context.

- **Backend:** Python 3.11+, FastAPI. **Frontend:** React 18 + TypeScript + Vite.
- **Providers:** OpenAI, Google Gemini, Anthropic Claude, OpenRouter, Groq, DeepSeek, Mistral, Together AI, plus one custom
  OpenAI-compatible endpoint (for example a local **Ollama**).
- **Status:** a working personal/local tool, not a hardened multi-user service. Read [Security](#security-and-privacy) before
  exposing it beyond your own computer.

## Contents
[Features](#features) · [Quick start](#quick-start) · [Using local Ollama](#using-local-ollama) · [Configuration](#configuration) ·
[How it works](#how-it-works) · [Model Lab](#model-lab) · [Spending controls](#spending-controls) · [Security and privacy](#security-and-privacy) ·
[Limitations](#limitations) · [Troubleshooting](#troubleshooting) · [Development](#development) · [API](#api-overview)

## Features

**Chat**
- A single window for everything; no task modes. Markdown with code blocks (each with a **Copy** button), generated images with
  **Download**, dark / light / system themes.
- **Attach** by button, drag-and-drop or paste: images, video, audio, PDFs, Word / Excel / PowerPoint / OpenDocument, code, CSV / JSON /
  logs, and zip / tar / tar.gz / gz archives (25 MB per file, 60 MB per message).
- **Copy** any message. **Edit** your own message and resend (later messages are replaced). **Regenerate** the last reply, optionally
  with a different model. **Retry** after a failure. Rename, search, delete and export chats to Markdown.
- **Output chip (Auto / Text / Image):** Auto makes an image when you ask for one ("generate an image of…") and answers in text
  otherwise. A message with an image attached is treated as "read this" unless you pick Image.

**Models**
- **Auto-pick:** chooses among every model your connected providers offer, from what they report and what has been proven to work.
  No vendor or model names are favoured.
- **Prefer free:** free models first, then unclassified, then paid. Nothing is excluded.
- **Safe fallback:** if a model fails for any reason, the next capable model is tried until one answers.
- **Manual choice** of provider and any model id, including models you add yourself.
- **Model Lab:** tests each API and model with real small requests to find out what it can do (see [Model Lab](#model-lab)).

**Local memory**
- History, attached files, generated images and per-chat notes ("Chat context") are stored in a local SQLite file.

**Dashboard**
- Request and token counts (provider-reported vs estimated), recent requests, a model catalog with per-model capability badges, free /
  paid classification, spending limits, provider detection.

## Quick start

Requirements: **Python 3.11+**, **Node.js 20+**, and an API key for at least one provider (or a local Ollama).

### 1. Backend

Windows (Command Prompt):
```bat
cd backend
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Windows (PowerShell): activate with `venv\Scripts\Activate.ps1`; the other lines are the same.
macOS / Linux:
```bash
cd backend
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Convenience scripts that activate the virtual environment for you: `scripts/start-backend.bat` and `scripts/start-backend.sh`.
Interactive API docs: <http://127.0.0.1:8000/docs>.

### 2. Frontend (second terminal)
```bash
cd frontend
npm install
npm run dev
```
Open <http://localhost:5173>.

### 3. Add a key
Click **Settings**, pick a provider, paste its API key and save. Keys go to your operating system's credential store (Windows
Credential Manager, macOS Keychain, Secret Service on Linux) through the `keyring` library, never to the browser. Alternatively set the
environment variables listed under [Configuration](#configuration). Then open **Dashboard → Refresh catalog** to load each provider's
current model list, and optionally run [Model Lab](#model-lab).

## Using local Ollama

Ollama exposes an OpenAI-compatible API, so it plugs into the custom-endpoint slot. Pull some models (`ollama pull qwen3:4b`) and make sure
Ollama is running, then start the backend with:

Command Prompt:
```bat
set LLMHUB_COMPATIBLE_BASE_URL=http://localhost:11434/v1
set LLMHUB_COMPATIBLE_LABEL=Ollama (local)
set LLMHUB_COMPATIBLE_API_KEY=ollama
set LLMHUB_COMPATIBLE_MODELS=qwen3:4b,gemma3:4b
uvicorn main:app --reload
```
PowerShell uses `$env:LLMHUB_COMPATIBLE_BASE_URL = "http://localhost:11434/v1"` and so on; macOS / Linux use `export`.
`LLMHUB_COMPATIBLE_API_KEY` can be any non-empty text (Ollama ignores it). `LLMHUB_COMPATIBLE_MODELS` is optional: the app asks
`/v1/models` for the list and uses this only as a fallback. Only **one** custom endpoint is supported at a time.

Then in the Dashboard mark the Ollama models **Free** (so *Prefer free* favours them) and run Model Lab on them: Ollama does not say which
models accept images, so the app learns it from the test. On a small GPU keep the context short (`OLLAMA_CONTEXT_LENGTH=4096`).

## Configuration

All settings are environment variables read by the backend at start-up (see `.env.example`). The backend does **not** load `.env` files
itself.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `TOGETHER_API_KEY` | Provider keys. Used only when no key is saved in the credential store. |
| `LLMHUB_COMPATIBLE_BASE_URL` / `_API_KEY` / `_LABEL` / `_MODELS` | One custom OpenAI-compatible endpoint (e.g. Ollama). |
| `LLMHUB_TELEMETRY_DB` | Path of the local SQLite file (default `~/.llm_hub/telemetry.sqlite3`). Holds chats, files, usage and capability results. |
| `VITE_API_BASE_URL` (frontend `.env`) | Backend URL, default `http://127.0.0.1:8000/api`. |

## How it works

### Attachments
Documents, spreadsheets, slides, code and logs are converted to text locally, so **every** text model can read them. A PDF with a text
layer also gets a text copy (needs `pypdf`), so a text-only model can still answer; scanned PDFs, images, audio and video are sent in the
provider's native format to models that accept them. Archives are opened **in memory**: nothing is written to disk and member names are only
labels (up to 300 files, 12 media files per archive).

### Auto-pick
For each request the app works out what it carries (image / PDF / audio / video), then considers every model your connected providers
currently list:
1. A model is **dropped** only when the *provider's own model list* says it can't take a needed input. OpenRouter's list includes
   this; Mistral and Anthropic may include some of it (the app reads it if present); OpenAI, Gemini and Groq don't say, so their models
   start as "unknown".
2. If nobody has said, the model is **unknown**: still tried, after confirmed ones. Real results are remembered. A success marks the
   model as accepting that input; a clear rejection demotes it to last place for 3 days, never removes it.
3. Order: not recently failing → (with *Prefer free*) free, then unclassified, then paid → confirmed input support → passed a Model
   Lab text test → context window large enough → providers interleaved, least-used-today first.
"Free" comes only from provider pricing data or what you set in the catalog.

### Safe fallback
Walks the same ordered list until a model answers. Any failure moves on: rate limit, server error, timeout, bad key, retired model,
rejected input, too-large request, empty reply (up to 12 attempts / 3 minutes; a bad key skips the rest of that provider). If no model accepts
your files, the request is re-sent as text only with a note naming what was left out. A model you picked yourself is tried first.
With Safe fallback off, a failure is shown as it is.

> **Cost trade-off:** retrying after a timeout can occasionally bill twice, and a fallback can use a different provider than the one
> you picked. Budget limits are checked per attempt.

### Local memory
For every request the server rebuilds the context for whichever model answers next, so switching model (or an automatic fallback) loses
nothing: the whole conversation; your *Chat context* notes; the extracted text of files attached earlier (newest first, 120k characters);
images / audio / video / scanned PDFs from the **last 2 turns that had them** (up to 12 MB); older media is named but not re-sent. Original
bytes are kept up to 200 MB per chat, after which the oldest bytes are dropped (their extracted text stays). This is a fixed rule, not an
AI-written summary, so it costs nothing and cannot invent facts; very long chats are still shortened to fit each model's context window.

### Image generation
Image **output** works for Google Gemini only (the only adapter that can carry an image back). Which Gemini models can draw is learned from
Model Lab and from real attempts; Gemini's model list doesn't say, so for models nobody has tried yet the adapter's own hint (an "image" in
the model id) only decides the order of the first attempts. If no model can, an automatically detected image request falls back to a text
answer with a note; a forced *Image* request reports the error.

## Model Lab

**Dashboard → Model Lab** sends small real requests to find out what each model can do. Only providers with a saved key can be tested.

| Test | How it's checked |
|---|---|
| Text | Reply "pong". |
| Code | Write a small Python function. |
| Image in | A generated coloured square; the model must name the colour. A model that ignores the image fails. |
| PDF in | A generated PDF with a secret code; the model must repeat it. |
| Audio in | A one-second tone; checks that the input is accepted (what it hears can't be verified). |
| Video in | A one-second clip made with **ffmpeg** if installed, or upload a short sample. Accepted-only, like audio. |
| Image out | Asks for a picture (Gemini only). |

Run one model with **Test**, or a provider with **Test up to N models** (stops if the key is rejected or a spending limit is hit).
Rate limits, server errors and timeouts are shown as inconclusive and are **not** recorded against a model. Results appear as badges in
the catalog and feed Auto-pick and Safe fallback. Probes cost a few tokens each and respect your spending limit.

## Spending controls

`GET /api/budgets` and `PUT /api/budgets` read and set limits (stored in the local database). Prices are user-supplied (USD per million
tokens) and intentionally not guessed:

```json
{
  "daily_limit_usd": 2.0,
  "monthly_limit_usd": 25.0,
  "provider_limits_usd": {"openai": 10.0, "gemini": 10.0},
  "pricing": {
    "openai": {"gpt-4.1-mini": {"input_per_million_usd": 0.4, "output_per_million_usd": 1.6}}
  }
}
```
A request is blocked only if a configured price makes its *estimate* exceed a limit; a model without a price is never blocked (there is
nothing meaningful to estimate). The estimate is character-based and reserves 2,048 output tokens, so it is a guard rail, not a guarantee.
Provider quota / rate-limit data is shown as unavailable unless an adapter can verify it. Treat your provider's dashboard as the source of truth.

## Security and privacy

- **Local by design.** The API has **no authentication** and its CORS policy allows only the Vite dev origins
  (`localhost:5173`, `127.0.0.1:5173`). Bind it to `127.0.0.1` (as in the commands above) and don't expose it to a network without adding
  authentication and a production CORS configuration.
- **API keys** are stored in the OS credential store, never in the browser or the database, and are never returned by the API.
- **Your data is stored unencrypted** in the local SQLite file (default `~/.llm_hub/telemetry.sqlite3`): chat text, the files you
  attached and generated images. Anyone with access to your user account can read it. Delete a chat in the sidebar to remove its messages
  and stored files, or delete the file to remove everything.
- **What leaves your machine:** each request goes to the provider you (or Auto-pick / Safe fallback) selected, including the context
  described under *Local memory*. Fallback may send the same context to a second provider.
- Don't commit `.env` files or the database. `.gitignore` excludes them.

## Limitations

- Replies arrive all at once; there is no token streaming yet.
- Video **generation** isn't implemented; a request for one gets a normal text answer. Image generation is Gemini-only.
- Output-type detection ("make an image") reads English wording; use the Auto / Text / Image chip to override it.
- Provider catalogs and pricing change. Model lists are fetched live where a provider offers a list endpoint; otherwise a small starter list
  is used and may contain outdated ids. Token counts are provider-reported when available, otherwise estimated.
- One custom OpenAI-compatible endpoint at a time. Ollama is used through that endpoint.
- Model Lab's audio and video checks only verify that the input is accepted.
- No multi-user support, accounts or sync between computers. Export chats if you need a copy.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `The filename, directory name, or volume label syntax is incorrect` when setting variables | You're in Command Prompt, not PowerShell: use `set NAME=value` (no spaces, no quotes). |
| "Backend offline" in the UI | Start the backend (`uvicorn main:app …`) and check <http://127.0.0.1:8000/api/health>. |
| Empty model list for a provider | Save its key in Settings, then Dashboard → Refresh catalog. |
| Ollama models missing | Is Ollama running (<http://localhost:11434/api/tags>)? Set `LLMHUB_COMPATIBLE_MODELS` as a fallback. |
| PDF / spreadsheet content not read by text models | `pip install pypdf openpyxl` (both are in `requirements.txt`). |
| Video test is skipped | Install **ffmpeg** or upload a sample clip in Model Lab. |
| Key can't be saved on Linux | `keyring` needs a Secret Service; use the environment variables instead. |

## Development

```bash
cd backend && python -m unittest discover -s tests -v   # 145 tests, no network or API keys needed
cd frontend && npm run build                            # type-checks and builds
```
The tests use throwaway databases (they refuse to run destructive helpers against anything outside the temp directory) and stand-ins
for FastAPI and the provider SDKs when those aren't installed. GitHub Actions (`.github/workflows/ci.yml`) runs both commands. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module map and request flow.

```
backend/
  main.py                FastAPI app and endpoints
  llm_hub/
    auto_router.py       Auto-pick ordering, Safe fallback, text-only last resort
    capabilities.py      what a model can do: provider metadata + observed results; failure classification
    attachments.py       file type detection, text extraction, safe archive handling
    conversations.py     local chat store and per-model context building
    probes.py            Model Lab test assets and checks
    intent.py            text-vs-image output detection
    catalog.py           live model lists from providers
    model_registry.py    catalog enrichment, billing, custom models, capability memory
    smart_router.py      context packing; legacy single-model routing (image path)
    task_routing.py      legacy ranking (media tasks, cost estimates)
    providers/           one adapter per API family + neutral→wire content converters
    telemetry.py, vault.py   usage metrics + budgets; OS credential store
  tests/                 145 tests
frontend/src/
  App.tsx                shell: sidebar, settings, themes
  components/            Chat, MessageView, ModelPicker, Markdown, ModelLab
  Dashboard.tsx          usage, catalog, Model Lab host
  api.ts, types.ts, util.ts
scripts/                 start-backend.bat / .sh
```

## API overview

Interactive documentation is served at `/docs`. Main routes (all under `/api`):

| Route | Purpose |
|---|---|
| `POST /chat` | One chat turn: message + attachments, optional `chat_id`, `edit_message_id`, `regenerate`, `output` (auto/text/image), `auto`, `prefer_free`, `safe_fallback`, `provider_id`/`model_id` |
| `GET/POST /chats`, `GET/PUT/DELETE /chats/{id}`, `GET /chats/{id}/files/{file_id}`, `POST /chats/import` | Local conversation store |
| `POST /probe` | Model Lab: test one model |
| `GET /providers`, `POST /catalog/refresh` | Model catalog with capability badges |
| `POST /models/custom`, `DELETE /models/custom/{provider}/{model}`, `PUT /models/billing`, `POST /models/test-image` | Manage and classify models |
| `GET /route-preview` | Which model Auto-pick would try first |
| `GET /dashboard`, `GET /detect` | Usage metrics; detect configured providers and a local Ollama |
| `POST /keys`, `DELETE /keys/{provider}` | Save / remove an API key in the credential store |
| `GET/PUT /budgets` | Spending limits |
| `GET /health` | Liveness |
| `POST /chat` with `messages` (no `chat_id`) | Stateless use: send the history yourself |
