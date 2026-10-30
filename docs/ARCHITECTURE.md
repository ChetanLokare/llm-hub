# Architecture

LLM Hub is a local FastAPI backend plus a React single-page app. The backend owns all provider access, keys, history and routing
decisions; the browser only renders and sends user actions.

```
 Browser (React)  ──HTTP/JSON──▶  FastAPI (main.py)  ──▶  provider adapters  ──▶  OpenAI / Gemini / Anthropic / OpenAI-compatible
                                        │
                                        ├── SQLite (~/.llm_hub/telemetry.sqlite3): chats, files, usage, budgets, capability memory
                                        └── OS credential store (keyring): API keys
```

## Request flow: `POST /api/chat`

1. **Prepare attachments** (`attachments.py`). Base64 uploads are decoded and typed from their *bytes* (not just the file name).
   Documents / code / logs become text; archives are opened in memory (`zipfile` / `tarfile` / `gzip`, never extracted to disk);
   images, audio, video and PDFs stay native. Limits: 25 MB per file, 60 MB per message, 300 archive members, 12 media files per archive,
   nesting depth 2.
2. **Store the turn** (`conversations.py`). The user's message and files are saved *before* any model is called, so a failed send is
   never lost. `edit_message_id` truncates history from that message; `regenerate` removes replies after the last user message.
3. **Build context** (`conversations.build_history`). Full history; extracted text of earlier files (120k chars, newest first);
   native media only from the last 2 turns that had any (12 MB); pinned chat context goes into the system prompt.
4. **Decide the output type** (`intent.py`). `output` = `text` / `image` forces it; `auto` uses a keyword heuristic over the message
   and treats a message with attached media as "read this".
5. **Image path** (`main._image_chat`) or **text path** (`main._text_chat` → `auto_router.run_chat`).
6. **Persist the answer** (including generated images) and return it with routing details (`routing_reason`, `fallback_reason`,
   `attempts`, `degraded`, `skipped_attachments`).

## Auto-pick and Safe fallback (`auto_router.py`)

`plan_attempts` builds an ordered list of `Attempt`s from every model every keyed provider lists:

| Tier key (ascending) | Meaning |
|---|---|
| preferred | The model chosen manually (when Auto-pick is off) |
| bad | Recently failed (in-memory `Health` cooldown) **or** failed a Model Lab text test |
| billing | Only when Prefer free is on: free < unknown < paid |
| confirmed | Needed inputs confirmed `yes` < untested < recently rejected |
| proven | Passed a Model Lab text test < untested |
| fits | Published context window large enough for the estimated request |

Within a tier, each provider's models are first sorted by larger published context window, then randomly. The tier is then
interleaved: one model per provider per round, with providers ordered by tokens used today, then by last success, then randomly.
**Nothing is ordered by vendor or model name.** A model is *removed* only if the provider's own model list says it can't accept a needed
input; a model you pick manually is never removed. Everything else is demotion.

`run_chat` executes the list. Any exception moves to the next attempt when Safe fallback is on (up to 12 attempts / 180 s; an auth failure
skips the rest of that provider; an empty reply counts as a failure). Outcomes are recorded: success on a request carrying media records
`input:<kind> = yes`; a clear modality rejection records `no` (3-day TTL). If every attempt fails for a request with native media, it
is retried once as text only (`Prepared.text_only`).

## Capability memory (`capabilities.py`, `model_registry.py`)

Capabilities are tri-state per model and input kind, resolved in this order:
`observed result` (success, or rejection within 3 days) → `provider model list` (`declared_modalities`) → `user-declared` →
`unknown`. Statuses: `yes`, `no` (provider says so; the only status that removes a model), `unlikely` (observed rejection; demotes),
`unknown`. Observations live in the `capability_checks` table under keys such as `input:image`, `works:text`, `works:code`,
`image_generation`. Failures are classified by `failure_kind` (rate_limit, auth, not_found, too_large, modality, server, bad_request,
unknown); transient kinds are never recorded as capability evidence.

## Model Lab (`probes.py`, `POST /api/probe`)

Generates test assets without dependencies (PNG via `zlib`, a valid one-page PDF, a WAV tone; an MP4 only if `ffmpeg` exists) and runs
one case per requested kind against **one** model per call. Verifiable cases (image colour, PDF secret code, `pong`, a function) must be
answered correctly to count; a model that answers without using the file is recorded as *not* accepting it. Inconclusive results are not
recorded. The frontend loops over models one request at a time so progress is visible and a run can be stopped.

## Providers (`providers/`)

`ProviderAdapter` subclasses implement `chat(model_id, messages, api_key, system_prompt)`. Message content is either a string or a list of
neutral blocks (`{"type": "text"}` / `{"type": "media", "kind", "mime_type", "data"}`); `providers/parts.py` converts these to each wire
format and raises `UnsupportedModality` when an API can't carry a kind of input (which the router treats as "try another model"). The
OpenAI-compatible adapter serves OpenRouter, Groq, DeepSeek, Mistral, Together and the custom endpoint (Ollama).
Model lists come from `catalog.py` (live, cached 15 minutes), merged with custom models and your billing overrides.

## Data model (SQLite)

| Table | Content |
|---|---|
| `chats`, `chat_messages`, `chat_files` | Conversations; messages (with generated images as JSON); attachments (extracted text and, up to the cap, bytes) |
| `capability_checks` | Observed capability results with evidence and timestamps |
| usage / budget / pricing tables | Telemetry and spending limits (`telemetry.py`) |

## Frontend

`App.tsx` holds the shell (sidebar chat list, settings, theme, catalog). `components/Chat.tsx` owns one conversation: it optimistically
shows your message, calls `/chat`, then reloads the conversation from the server so the server stays the source of truth.
`ModelPicker` edits persisted preferences (`localStorage`: only UI preferences, never chats or keys). `ModelLab` (inside `Dashboard.tsx`)
drives `/probe`. Types are in `types.ts`; the API base URL comes from `VITE_API_BASE_URL`.

## Testing approach

Everything routing-related is SDK-free and unit-tested with fake providers: ordering and bias tests (many seeds), every failure kind,
learning from results, context across model switches, edit/regenerate, image fallback, Model Lab verdicts, archive safety (monkey-patched
filesystem), key redaction. `tests/_stubs.py` supplies stand-ins for FastAPI / SDKs when they aren't installed, and forces a throwaway
database so tests can never touch real data.

## Known design limits

No streaming; single custom endpoint slot; Gemini-only image output; English-only intent detection; no authentication (local use);
conversation data is stored unencrypted.
