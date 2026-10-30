
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_hub import conversations as conv
from llm_hub import model_registry as registry
from llm_hub.intent import detect_output_intent
from llm_hub.probes import ALL_KINDS as PROBE_KINDS, ProbeResult, run_probe
from llm_hub.attachments import AttachmentError, Prepared, prepare_attachments
from llm_hub.auto_router import (
    AllAttemptsFailed,
    BudgetBlocked,
    MissingKey,
    NoCandidates,
    no_candidates_message,
    plan_attempts,
    run_chat,
)
from llm_hub.capabilities import MEDIA_KINDS, redact
from llm_hub.catalog import get_catalog
from llm_hub.router import build_providers
from llm_hub.smart_router import SmartRouter, _unpack, pack_context
from llm_hub.task_routing import (
    MEDIA_TASKS,
    TASKS,
    Selection,
    choose_model,
    estimate_input_tokens,
    live_provider_views,
)
from llm_hub.telemetry import (
    check_budget,
    dashboard_snapshot,
    estimate_configured_cost,
    get_spend_config,
    record_usage,
    save_spend_config,
)
from llm_hub.vault import delete_secret, get_secret, save_secret


# ---------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------

app = FastAPI(
    title="LLM Hub API",
    version="0.2.0",
    description="Multi-provider LLM routing, model discovery, and usage tracking.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

providers = build_providers()

ENV_KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "compatible": "LLMHUB_COMPATIBLE_API_KEY",
}


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def get_provider_key(provider_id: str) -> str | None:
    """Get a provider API key from the credential store or environment."""
    stored_key = get_secret(provider_id)
    if stored_key:
        return stored_key

    environment_name = ENV_KEY_NAMES.get(provider_id)
    if not environment_name:
        return None

    return os.environ.get(environment_name, "").strip() or None


def get_provider_ids() -> set[str]:
    """Return IDs for providers registered in the application."""
    return {provider.provider_id for provider in providers}


# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------

class AttachmentIn(BaseModel):
    """A file sent with the message. ``data`` is base64 (a data: URL prefix is tolerated)."""
    name: str = Field(max_length=200)
    mime_type: str = ""
    data: str


class ChatRequest(BaseModel):
    """One turn of the single chat window.

    Stateful use (what the app does): send ``message`` (+ ``attachments``) with a ``chat_id``; the server keeps the
    history, files and pinned context locally and rebuilds them for whichever model answers. ``edit_message_id``
    replaces one of your earlier messages and re-answers from there; ``regenerate`` re-answers your last message.
    ``messages`` (stateless, full history in the request) is still accepted for simple API use.
    """
    task: str = "General chat"          # legacy label; only "Image generation" still forces image output
    provider_id: str | None = None
    model_id: str | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)
    message: str = ""
    chat_id: str | None = None
    edit_message_id: int | None = None
    regenerate: bool = False
    keep_files: bool = True
    output: str = "auto"                # auto | text | image
    safe_fallback: bool = False
    auto: bool = False
    prefer_free: bool = False
    context: str = ""
    attachments: list[AttachmentIn] = Field(default_factory=list)


class ChatCreateRequest(BaseModel):
    id: str | None = None
    title: str = "New chat"
    context: str = ""


class ChatImportRequest(BaseModel):
    title: str = "Imported chat"
    context: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)


class ChatUpdateRequest(BaseModel):
    title: str | None = None
    context: str | None = None


class ProbeRequest(BaseModel):
    provider_id: str
    model_id: str = Field(min_length=1, max_length=200)
    kinds: list[str] = Field(default_factory=lambda: list(PROBE_KINDS))
    video_sample: AttachmentIn | None = None


class CustomModelRequest(BaseModel):
    provider_id: str
    model_id: str = Field(min_length=1, max_length=200)
    display_name: str = Field(default="", max_length=120)
    capabilities: list[str] = Field(default_factory=lambda: ["text"])
    context_window: int | None = Field(default=None, ge=1000, le=100_000_000)
    billing: str = "unknown"


class ImageTestRequest(BaseModel):
    provider_id: str
    model_id: str = Field(min_length=1, max_length=200)


class BillingRequest(BaseModel):
    provider_id: str
    model_id: str = Field(min_length=1, max_length=200)
    billing: str  # free | paid | auto


class KeyRequest(BaseModel):
    provider_id: str
    api_key: str


class BudgetConfigRequest(BaseModel):
    daily_limit_usd: float | None = Field(default=None, ge=0)
    monthly_limit_usd: float | None = Field(default=None, ge=0)
    provider_limits_usd: dict[str, float] = Field(default_factory=dict)
    pricing: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------
# Health and model catalog
# ---------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    """Confirm that the API is running."""
    return {
        "status": "ok",
        "service": "LLM Hub API",
    }


@app.get("/api/providers")
def list_providers(refresh: bool = False) -> Any:
    """
    Return provider and model information.

    The catalog module attempts live model discovery where supported.
    When discovery is unavailable, it should return its local fallback
    catalog and indicate the source in its response.
    """
    try:
        return registry.enrich_catalog(
            get_catalog(
                providers,
                get_provider_key,
                refresh=refresh,
            )
        )
    except Exception as exc:
        # Keep the error visible so discovery problems can be diagnosed.
        raise HTTPException(
            status_code=502,
            detail=f"Could not load the provider catalog: {redact(str(exc))}",
        ) from exc


@app.post("/api/catalog/refresh")
def refresh_model_catalog() -> Any:
    """Force a model catalog refresh."""
    try:
        catalog = registry.enrich_catalog(
            get_catalog(
                providers,
                get_provider_key,
                refresh=True,
            )
        )
        return {
            "refreshed": True,
            "providers": catalog,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not refresh the model catalog: {redact(str(exc))}",
        ) from exc


# ---------------------------------------------------------------------
# Added models, free/paid classification, auto-pick preview
# ---------------------------------------------------------------------

def _routing_view() -> list:
    """Providers as routing should see them: live model lists, your added models, billing info."""
    try:
        catalog = get_catalog(providers, get_provider_key)
    except Exception:
        catalog = []
    try:
        views = live_provider_views(providers, catalog)
    except Exception:
        views = providers
    return registry.routing_providers(views, catalog)


@app.post("/api/models/custom")
def add_custom_model(request: CustomModelRequest) -> dict[str, Any]:
    """Add a model the provider list does not show yet (for example a newly released model)."""
    if request.provider_id not in get_provider_ids():
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider_id}'.")
    try:
        registry.add_custom_model(request.provider_id, request.model_id, request.display_name,
                                  request.capabilities, request.context_window, request.billing)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True}


@app.delete("/api/models/custom/{provider_id}/{model_id:path}")
def delete_custom_model(provider_id: str, model_id: str) -> dict[str, Any]:
    return {"deleted": registry.delete_custom_model(provider_id, model_id)}


@app.put("/api/models/billing")
def set_model_billing(request: BillingRequest) -> dict[str, Any]:
    """Mark a model as free or paid (or 'auto' to go back to automatic classification)."""
    if request.provider_id not in get_provider_ids():
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider_id}'.")
    try:
        registry.set_billing_override(request.provider_id, request.model_id, request.billing)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True}


@app.get("/api/route-preview")
def route_preview(task: str = "General chat", prefer_free: bool = False, modalities: str = "") -> dict[str, Any]:
    """Show which model Auto-pick would try first right now, and the next ones Safe fallback would try.

    ``modalities`` is a comma list of the input kinds attached to the message (image, pdf, audio, video).
    """
    required = {m for m in (x.strip() for x in modalities.split(",")) if m in MEDIA_KINDS}
    views = _routing_view()

    def has_key(pid: str) -> bool:
        return bool(get_provider_key(pid))

    plan = plan_attempts(views, required, set(), 0, registry.get_capability_checks(), has_key=has_key,
                         prefer_free=prefer_free, load=registry.provider_load_today())
    result: dict[str, Any] = {"task": task, "prefer_free": prefer_free, "provider_id": None, "alternatives": [],
                              "reason": no_candidates_message(views, required, has_key)}
    if plan:
        from llm_hub.auto_router import explain
        top = plan[0]
        result.update(provider_id=top.provider.provider_id, provider_name=top.provider.display_name,
                      model_id=top.model.model_id, reason=explain(top, required, True))
    result["alternatives"] = [
        {"provider_id": a.provider.provider_id, "provider_name": a.provider.display_name, "model_id": a.model.model_id,
         "billing": a.billing, "input_support": {k: v[0] for k, v in a.statuses.items()}}
        for a in plan[:5]
    ]
    return result


# ---------------------------------------------------------------------
# Dashboard and integration detection
# ---------------------------------------------------------------------

@app.get("/api/dashboard")
def dashboard() -> Any:
    """Return usage and dashboard metrics."""
    return dashboard_snapshot(
        providers,
        get_provider_key,
    )


@app.get("/api/detect")
def detect_integrations() -> dict[str, Any]:
    """
    Detect configured credentials and check the local Ollama endpoint.

    This endpoint does not scan arbitrary files or expose secret values.
    """
    detected_providers = []

    for provider in providers:
        provider_id = provider.provider_id
        environment_name = ENV_KEY_NAMES.get(provider_id)

        environment_present = bool(
            os.environ.get(environment_name, "").strip()
        ) if environment_name else False

        credential_store_present = bool(get_secret(provider_id))

        if environment_present:
            source = "environment variable"
        elif credential_store_present:
            source = "OS credential store"
        else:
            source = "not detected"

        detected_providers.append(
            {
                "provider_id": provider_id,
                "display_name": provider.display_name,
                "environment_variable_present": environment_present,
                "credential_store_present": credential_store_present,
                "detected": (
                    environment_present or credential_store_present
                ),
                "source": source,
            }
        )

    ollama_endpoint = "http://127.0.0.1:11434"
    ollama_models = []
    ollama_detected = False

    try:
        with urllib.request.urlopen(
            f"{ollama_endpoint}/api/tags",
            timeout=1.2,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        ollama_detected = True
        ollama_models = [
            {
                "name": model.get("name"),
                "size": model.get("size"),
            }
            for model in data.get("models", [])
        ]

    except Exception:
        # Ollama is optional; failure to detect it is not an API error.
        pass

    return {
        "providers": detected_providers,
        "local_services": [
            {
                "name": "Ollama",
                "provider_id": "ollama",
                "detected": ollama_detected,
                "endpoint": ollama_endpoint,
                "models": ollama_models,
                "note": (
                    "Detection only; the Ollama chat adapter must be "
                    "enabled separately."
                ),
            }
        ],
    }


# ---------------------------------------------------------------------
# Task routing
# ---------------------------------------------------------------------

@app.get("/api/tasks")
def list_tasks() -> list[str]:
    """Return supported task categories."""
    return list(TASKS.keys())


# ---------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------

@app.post("/api/keys")
def set_key(request: KeyRequest) -> dict[str, Any]:
    """Save a provider API key in the configured credential store."""
    if request.provider_id not in get_provider_ids():
        raise HTTPException(
            status_code=400,
            detail="Unknown provider.",
        )

    api_key = request.api_key.strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key cannot be empty.",
        )

    try:
        save_secret(request.provider_id, api_key)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save API key: {exc}",
        ) from exc

    return {
        "saved": True,
        "provider_id": request.provider_id,
    }


@app.delete("/api/keys/{provider_id}")
def remove_key(provider_id: str) -> dict[str, Any]:
    """Remove a provider API key from the credential store."""
    if provider_id not in get_provider_ids():
        raise HTTPException(
            status_code=404,
            detail="Unknown provider.",
        )

    try:
        delete_secret(provider_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete API key: {exc}",
        ) from exc

    return {
        "deleted": True,
        "provider_id": provider_id,
    }


# ---------------------------------------------------------------------
# Spending limits and model pricing
# ---------------------------------------------------------------------

@app.get("/api/budgets")
def read_budgets() -> dict[str, Any]:
    """Return the configured limits and recorded spending totals."""
    from llm_hub.telemetry import spend_totals

    return {
        "config": get_spend_config(),
        "totals": spend_totals(),
        "enforcement_note": (
            "Preflight spending checks require configured input and output "
            "prices for the selected provider/model. Cost calculations are "
            "estimates, not provider invoices."
        ),
    }


@app.put("/api/budgets")
def update_budgets(request: BudgetConfigRequest) -> dict[str, Any]:
    """Validate and save spending limits and model pricing."""
    allowed_provider_ids = get_provider_ids()

    for provider_id, limit in request.provider_limits_usd.items():
        if provider_id not in allowed_provider_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Spending limit contains an unknown provider: "
                    f"{provider_id}"
                ),
            )

        if limit < 0:
            raise HTTPException(
                status_code=400,
                detail="Provider spending limits cannot be negative.",
            )

    for provider_id, models in request.pricing.items():
        if provider_id not in allowed_provider_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider in pricing: {provider_id}",
            )

        for model_id, rates in models.items():
            required_fields = {
                "input_per_million_usd",
                "output_per_million_usd",
            }

            if not required_fields.issubset(rates):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Pricing for {provider_id}/{model_id} must include "
                        "input_per_million_usd and output_per_million_usd."
                    ),
                )

            if any(
                not isinstance(rate, (int, float))
                or rate < 0
                for rate in rates.values()
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Pricing rates must be non-negative numbers.",
                )

    config = request.model_dump()

    try:
        save_spend_config(config)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save spending configuration: {exc}",
        ) from exc

    return {
        "saved": True,
        "config": config,
    }


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

# LLM Hub's chat adapters return text only. Without this note, models often say "here is your image"
# and then show nothing. Remove it once a real image/video generation adapter is connected.
TEXT_ONLY_NOTE = (
    "You are being used through LLM Hub's text chat. This chat cannot create, display or attach images, "
    "audio or video, and it is not connected to any image or video generator. If the user asks for an image "
    "or video, say plainly that you cannot produce one here and offer to write a detailed prompt they can "
    "paste into an image or video generator. Never claim to have generated or attached media."
)


def _build_payload(request: ChatRequest, result: Any, routing_reason: str, budget_config: dict[str, Any],
                   estimated_input_tokens: int, budget_check: dict[str, Any]) -> dict[str, Any]:
    """Record usage and build the JSON response. ``result`` needs answer, usage, provider, model_id,
    context_note, fallback_reason and images (RouteResult and ChatOutcome both qualify)."""
    usage = result.usage

    reported_input_tokens = getattr(
        usage,
        "input_tokens",
        None,
    )
    reported_output_tokens = getattr(
        usage,
        "output_tokens",
        None,
    )

    estimated_output_tokens = (
        max(1, (len(result.answer) + 3) // 4)
        if result.answer
        else 0
    )

    input_tokens_for_cost = (
        reported_input_tokens
        or getattr(usage, "estimated_input_tokens", None)
        or estimated_input_tokens
    )

    output_tokens_for_cost = (
        reported_output_tokens
        or estimated_output_tokens
    )

    request_cost = estimate_configured_cost(
        result.provider.provider_id,
        result.model_id,
        input_tokens_for_cost,
        output_tokens_for_cost,
        budget_config,
    )

    record_usage(
        result.provider.provider_id,
        result.model_id,
        request.task,
        usage,
        request_cost,
        estimated_output_tokens,
    )

    images = list(getattr(result, "images", None) or [])
    answer = result.answer or ("" if images else
                               "The model returned no content. It may have been blocked by a safety filter.")
    return {
        "answer": answer,
        "images": images,
        "provider_id": result.provider.provider_id,
        "provider_name": result.provider.display_name,
        "model_id": result.model_id,
        "context_note": result.context_note,
        "fallback_reason": result.fallback_reason,
        "routing_reason": routing_reason,
        "usage": {
            "input_tokens": reported_input_tokens,
            "output_tokens": reported_output_tokens,
            "total_tokens": getattr(
                usage,
                "total_tokens",
                None,
            ),
            "source": getattr(
                usage,
                "source",
                "unknown",
            ),
            "estimated_cost_usd": request_cost,
            "budget_preflight": budget_check,
        },
    }


def _execute_chat(request: ChatRequest, selected: Selection, system_prompt: str, want_images: bool = False) -> dict[str, Any]:
    """Check the key and budget, send the request, record usage and build the response."""
    provider_id = selected.provider.provider_id

    if not get_provider_key(provider_id):
        raise HTTPException(
            status_code=400,
            detail=(
                f"No API key saved for {selected.provider.display_name}. "
                "Configure it in Settings."
            ),
        )

    budget_config = get_spend_config()

    estimated_input_tokens = (
        estimate_input_tokens(request.messages)
        + estimate_input_tokens(
            [{"content": request.context}]
        )
    )

    # Reserve an estimated output allowance before sending the request.
    budget_check = check_budget(
        provider_id,
        selected.model_id,
        estimated_input_tokens,
        2048,
        budget_config,
    )

    if not budget_check.get("allowed", False):
        raise HTTPException(
            status_code=402,
            detail={
                "message": budget_check.get(
                    "reason",
                    "The configured spending limit would be exceeded.",
                ),
                "budget": budget_check,
            },
        )

    try:
        result = SmartRouter(
            providers,
            get_provider_key,
            request.safe_fallback,
        ).run(
            selected.provider,
            selected.model_id,
            request.messages,
            system_prompt=system_prompt,
            want_images=want_images,
        )

        return _build_payload(request, result, selected.reason, budget_config, estimated_input_tokens, budget_check)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model request failed: {redact(str(exc))}",
        ) from exc


# ---------------------------------------------------------------------
# Conversations (local memory)
# ---------------------------------------------------------------------

@app.get("/api/chats")
def list_chats(q: str = "") -> list[dict[str, Any]]:
    return conv.list_chats(q)


@app.post("/api/chats")
def create_chat_endpoint(request: ChatCreateRequest) -> dict[str, Any]:
    row = conv.create_chat(request.id, request.title)
    if request.context:
        row = conv.update_chat(row["id"], context=request.context)
    return row


@app.post("/api/chats/import")
def import_chat(request: ChatImportRequest) -> dict[str, Any]:
    """One-time move of a chat saved by the earlier browser-only version into local server storage."""
    row = conv.create_chat(None, request.title[:120] or "Imported chat")
    for m in request.messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            conv.add_message(row["id"], m["role"], m["content"][:200_000],
                             provider=m.get("provider"), model_id=m.get("model"))
    if request.context:
        conv.update_chat(row["id"], context=request.context)
    return {"id": row["id"]}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str) -> dict[str, Any]:
    try:
        chat_row = conv.get_chat(chat_id)
    except conv.ChatNotFound:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {**chat_row, "messages": conv.messages(chat_id)}


@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: str, request: ChatUpdateRequest) -> dict[str, Any]:
    try:
        return conv.update_chat(chat_id, request.title, request.context)
    except conv.ChatNotFound:
        raise HTTPException(status_code=404, detail="Chat not found.")


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str) -> dict[str, Any]:
    conv.delete_chat(chat_id)
    return {"deleted": chat_id}


INLINE_KINDS = {"image", "audio", "video", "pdf"}


@app.get("/api/chats/{chat_id}/files/{file_id}")
def get_chat_file(chat_id: str, file_id: int) -> Any:
    blob = conv.file_blob(chat_id, file_id)
    if blob is None:
        raise HTTPException(status_code=404, detail="That file is no longer stored on this device.")
    data, mime, name = blob
    kind = "image" if mime.startswith("image/") else "audio" if mime.startswith("audio/") else \
        "video" if mime.startswith("video/") else "pdf" if mime == "application/pdf" else "other"
    safe = "".join(c for c in name if c.isalnum() or c in " ._-")[:100] or "file"
    disposition = "inline" if kind in INLINE_KINDS else "attachment"
    return Response(content=data, media_type=mime if kind in INLINE_KINDS else "application/octet-stream",
                    headers={"Content-Disposition": f'{disposition}; filename="{safe}"', "X-Content-Type-Options": "nosniff"})


# ---------------------------------------------------------------------
# Unified chat
# ---------------------------------------------------------------------

def _merge_prepared(first: Prepared, second: Prepared) -> Prepared:
    return Prepared([*first.parts, *second.parts], [*first.notes, *second.notes])


def _stateful_turn(request: ChatRequest, new_prepared: Prepared) -> dict[str, Any]:
    """Store this turn locally and return the context to send: history, earlier media, this message's files."""
    try:
        chat_row = conv.get_chat(request.chat_id) if request.chat_id else None
    except conv.ChatNotFound:
        chat_row = None
    if chat_row is None:
        chat_row = conv.create_chat(request.chat_id)
    cid = chat_row["id"]

    if request.regenerate:
        user_msgs = [m for m in conv.messages(cid, include_images=False) if m["role"] == "user"]
        if not user_msgs:
            raise HTTPException(status_code=400, detail="There is no message to answer again.")
        last = user_msgs[-1]
        conv.truncate_from(cid, last["id"] + 1)  # drop the reply (and anything after) but keep your message
        user_id, prepared = last["id"], conv.load_prepared(cid, last["id"])
    elif request.edit_message_id is not None:
        try:
            old = conv.get_message(cid, request.edit_message_id)
        except conv.ChatNotFound:
            raise HTTPException(status_code=404, detail="That message no longer exists.")
        if old["role"] != "user":
            raise HTTPException(status_code=400, detail="Only your own messages can be edited.")
        text = request.message.strip()
        if not text and not new_prepared.parts and not request.keep_files:
            raise HTTPException(status_code=400, detail="The edited message is empty.")
        kept = conv.load_prepared(cid, old["id"]) if request.keep_files else Prepared()
        conv.truncate_from(cid, old["id"])
        prepared = _merge_prepared(kept, new_prepared)
        user_id = conv.add_message(cid, "user", text or old["content"], prepared=prepared)
    else:
        text = request.message.strip() or ("Please look at the attached file(s)." if new_prepared.parts else "")
        if not text:
            raise HTTPException(status_code=400, detail="Type a message or attach a file.")
        first = not conv.messages(cid, include_images=False)
        user_id = conv.add_message(cid, "user", text, prepared=new_prepared)
        prepared = new_prepared
        if first and chat_row["title"] in ("", "New chat"):
            title = " ".join(text.split())[:48] + ("…" if len(" ".join(text.split())) > 48 else "")
            conv.update_chat(cid, title=title)

    history, earlier = conv.build_history(cid)
    context = conv.get_chat(cid)["context"]
    return {"chat_id": cid, "user_message_id": user_id, "history": history, "earlier": earlier,
            "prepared": prepared, "context": "\n\n".join(x for x in (context, request.context) if x)}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """One chat window for everything: text, images, documents, zips, audio, video in; text or images out.

    Manual: the selected model is tried first. Auto-pick: every model the connected providers report is a
    candidate, ordered from provider-reported capabilities and observed results (see auto_router.py).
    Safe fallback: any failure moves on to the next capable candidate until one answers.
    """
    stateful = bool(request.chat_id or request.message.strip() or request.regenerate or request.edit_message_id is not None)
    if not stateful and not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")

    try:
        new_prepared = prepare_attachments([a.model_dump() for a in request.attachments])
    except AttachmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.task in MEDIA_TASKS:  # legacy: video generation is unavailable unless a model declares it
        chosen = choose_model(_routing_view(), request.task, request.provider_id, request.model_id, auto=request.auto,
                              has_key=lambda pid: bool(get_provider_key(pid)), prefer_free=request.prefer_free)
        if not chosen.provider:
            raise HTTPException(status_code=400, detail=chosen.reason)

    if stateful:
        turn = _stateful_turn(request, new_prepared)
    else:
        turn = {"chat_id": None, "user_message_id": None, "history": request.messages, "earlier": {},
                "prepared": new_prepared, "context": request.context}
    history, prepared, context = turn["history"], turn["prepared"], turn["context"]
    last_user_text = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")

    mode = "image" if request.task == "Image generation" else request.output
    forced = mode in ("image", "text")
    if mode not in ("image", "text"):
        mode = detect_output_intent(last_user_text, bool(prepared.native_required()))
    payload: dict[str, Any] | None = None
    notice = ""
    if mode == "image":
        try:
            payload = _image_chat(request, history, context)
        except HTTPException as exc:
            if forced:
                raise
            notice = (f"I tried to create an image but couldn't ({exc.detail if isinstance(exc.detail, str) else 'no model available'}) "
                      "so this is a text answer.")
    if payload is None:
        payload = _text_chat(request, history, turn["earlier"], prepared, context)
        if notice:
            payload["capability_notice"] = notice
    payload["output_mode"] = "image" if payload.get("images") else "text"
    payload["detected_intent"] = mode

    if turn["chat_id"]:
        assistant_id = conv.add_message(
            turn["chat_id"], "assistant", payload["answer"], provider=payload.get("provider_name"),
            model_id=payload.get("model_id"),
            note=" ".join(x for x in (payload.get("routing_reason") if (request.auto or payload.get("fallback_reason")) else "",
                                      payload.get("fallback_reason") or "", payload.get("capability_notice") or "") if x) or None,
            images=payload.get("images"), meta={"had_images": bool(payload.get("images")),
                                                 "attachment_notes": payload.get("attachment_notes") or []})
        payload.update(chat_id=turn["chat_id"], user_message_id=turn["user_message_id"], assistant_message_id=assistant_id,
                       title=conv.get_chat(turn["chat_id"])["title"])
    return payload


def _text_chat(request: ChatRequest, history: list[dict], earlier: dict, prepared: Prepared, context: str) -> dict[str, Any]:
    def has_key(pid: str) -> bool:
        return bool(get_provider_key(pid))

    views = _routing_view()
    auto = request.auto or not request.provider_id
    selected: tuple[str, str] | None = None
    if not auto:
        provider = next((v for v in views if v.provider_id == request.provider_id), None)
        if provider is None:
            raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider_id}'.")
        model_id = request.model_id or next((m.model_id for m in provider.list_models()), None)
        if not model_id:
            raise HTTPException(status_code=400, detail=f"{provider.display_name} has no models configured.")
        selected = (provider.provider_id, model_id)

    system_prompt = TEXT_ONLY_NOTE + ("\n\n" + context if context else "")
    budget_config = get_spend_config()
    estimated_input_tokens = (
        estimate_input_tokens(history) + estimate_input_tokens([{"content": context}]) + prepared.estimated_tokens()
        + sum(e.estimated_tokens() for e in earlier.values())
    )
    last_budget: dict[str, Any] = {}

    def budget_gate(provider_id: str, model_id: str, _messages: list) -> None:
        check = check_budget(provider_id, model_id, estimated_input_tokens, 2048, budget_config)
        last_budget.clear()
        last_budget.update(check)
        if not check.get("allowed", False):
            raise BudgetBlocked({"message": check.get("reason", "The configured spending limit would be exceeded."),
                                 "budget": check})

    def send(provider: Any, model: Any, messages: list) -> tuple:
        key = get_provider_key(provider.provider_id)
        if not key:
            raise MissingKey(f"No API key saved for {provider.display_name}. Configure it in Settings.")
        packed, note = pack_context(messages, model.context_window, system_prompt=system_prompt)
        answer, usage, images = _unpack(provider.chat(model.model_id, packed, key, system_prompt=system_prompt))
        return answer, usage, images, note

    try:
        outcome = run_chat(
            views=views,
            history=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in history],
            prepared=prepared, earlier=earlier, selected=selected, auto=auto, safe_fallback=request.safe_fallback,
            prefer_free=request.prefer_free, has_key=has_key, send=send,
            checks=registry.get_capability_checks(), record_check=registry.record_capability_check,
            load=registry.provider_load_today(), budget_gate=budget_gate,
        )
    except BudgetBlocked as exc:
        raise HTTPException(status_code=402, detail=exc.detail) from exc
    except MissingKey as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoCandidates as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AllAttemptsFailed as exc:
        if isinstance(exc.last, BudgetBlocked):
            raise HTTPException(status_code=402, detail=exc.last.detail) from exc
        tried = " | ".join(f"{e['label']}: {e['error']}" for e in exc.log[:8])
        raise HTTPException(status_code=502,
                            detail=f"No model could complete the request after {len(exc.log)} attempt(s). {tried}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model request failed: {redact(str(exc))}") from exc

    if request.task == "General chat":
        request.task = "Chat"
    payload = _build_payload(request, outcome, outcome.routing_reason, budget_config, estimated_input_tokens, last_budget)
    payload.update(attempts=outcome.attempts, degraded=outcome.degraded, skipped_attachments=outcome.skipped,
                   attachment_notes=prepared.notes)
    return payload


# ---------------------------------------------------------------------
# Image generation: try the model, remember what happened
# ---------------------------------------------------------------------
# A model's own claim ("Yes, I can draw!") is not evidence, so the app asks for an image and records
# whether one actually came back. The result is stored per model and shown in the catalog.

IMAGE_NOTE = (
    "The user wants an image. Create it and return it. If you are not able to produce an image, say so "
    "plainly and do not claim to have created or attached one."
)
PROBE_PROMPT = "Draw a small, simple picture: a red circle on a white background."
NO_CONTENT_PREFIX = "The model returned no content"
NO_IMAGE_MODEL_MESSAGE = (
    "No connected model can produce images yet. Image output is currently supported for Google Gemini only: add a "
    "Gemini key, then run Model Lab (Dashboard) with 'Image output' to find which of its models can draw."
)


def _looks_like_rejected_modality(message: str) -> bool:
    """True only when the provider says it does not accept image output for this model."""
    text = message.lower()
    return "modalit" in text and any(
        phrase in text for phrase in ("not support", "unsupported", "invalid", "not enabled", "does not", "cannot")
    )


def _image_target(provider_id: str | None, model_id: str | None, auto: bool = False,
                  prefer_free: bool = False, retest: bool = False) -> tuple[Any, str, str]:
    """Pick and vet the provider/model for an image request. Raises HTTPException with a plain reason."""
    def has_key(pid: str) -> bool:
        return bool(get_provider_key(pid))

    supported = ", ".join(sorted(registry.IMAGE_OUTPUT_ADAPTERS))
    if auto or not provider_id:
        ranked = _image_candidates(prefer_free)
        if not ranked:
            raise HTTPException(status_code=400, detail=NO_IMAGE_MODEL_MESSAGE)
        return ranked[0]

    provider = next((p for p in providers if p.provider_id == provider_id), None)
    if provider is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{provider_id}'.")
    if not model_id:
        raise HTTPException(status_code=400, detail="Choose a model first.")
    if provider.provider_id not in registry.IMAGE_OUTPUT_ADAPTERS:
        raise HTTPException(
            status_code=400,
            detail=(f"{provider.display_name} can't be used for image generation in this app yet. That is a limit of "
                    f"the app, not necessarily of the model: its adapter here only returns text. Image output is "
                    f"currently supported for: {supported}."),
        )
    if not get_provider_key(provider.provider_id):
        raise HTTPException(status_code=400,
                            detail=f"No API key saved for {provider.display_name}. Configure it in Settings.")
    check = registry.get_capability_checks().get((provider.provider_id, model_id, "image_generation"))
    if check and not check["result"] and not retest:
        raise HTTPException(
            status_code=400,
            detail=(f"{model_id} was tested on {check['checked_at'][:10]}: {check['evidence']}. It can't generate "
                    "images, so the request was not sent. Pick another model, or use Test image output in the "
                    "catalog to check again."),
        )
    return provider, model_id, f"Using the selected {provider.display_name} / {model_id}."


def _try_image(provider: Any, model_id: str, messages: list[dict[str, str]], context: str, reason: str):
    """Ask the model for an image and record the outcome.

    Returns (verdict, payload): verified | text_only | rejected | inconclusive.
    Errors that say nothing about capability (bad key, rate limit, network) are raised and not recorded.
    """
    request = ChatRequest(task="Image generation", provider_id=provider.provider_id, model_id=model_id,
                          messages=messages, context=context)
    selected = Selection(provider, model_id, reason)
    system_prompt = IMAGE_NOTE + ("\n\n" + context if context else "")
    pid = provider.provider_id
    try:
        out = _execute_chat(request, selected, system_prompt, want_images=True)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        if _looks_like_rejected_modality(detail):
            registry.record_capability_check(pid, model_id, "image_generation", False,
                                             "The provider rejected an image request for this model")
            return "rejected", detail
        raise
    if out["images"]:
        registry.record_capability_check(pid, model_id, "image_generation", True, "Returned an image")
        return "verified", out
    if out["answer"].startswith(NO_CONTENT_PREFIX):
        return "inconclusive", out  # most likely a safety block; says nothing about capability
    registry.record_capability_check(pid, model_id, "image_generation", False, "Replied with text only, no image")
    return "text_only", out


def _image_candidates(prefer_free: bool = False, rng: Any = None) -> list[tuple[Any, str, str]]:
    """Every (provider, model_id, reason) that could be asked for an image, best evidence first.

    Only providers whose adapter here can carry an image back are considered. Order: models that already returned an
    image, then ones you marked image-capable, then untested ones (the app's Gemini adapter hints at image-output
    models by id, which only orders models nobody has tried yet), and models that failed an image test go last,
    never out. With Prefer free, free then unknown then paid comes first.
    """
    import random
    from llm_hub.auto_router import BILLING_RANK
    from llm_hub.providers.media import is_image_output_model

    rng = rng or random.Random()
    checks = registry.get_capability_checks()
    views = _routing_view()
    try:
        catalog = {e.get("provider_id"): e.get("models", []) for e in get_catalog(providers, get_provider_key)}
    except Exception:
        catalog = {}
    ranked: list[tuple[tuple, Any, str]] = []
    for provider in providers:
        pid = provider.provider_id
        if pid not in registry.IMAGE_OUTPUT_ADAPTERS or not get_provider_key(pid):
            continue
        billing = {m.model_id: m.billing for v in views if v.provider_id == pid for m in v.list_models()}
        declared = {m.model_id for v in views if v.provider_id == pid for m in v.list_models() if "image_generation" in m.capabilities}
        ids = list(dict.fromkeys([*billing, *(m.get("model_id") for m in catalog.get(pid, []) if m.get("model_id"))]))
        for mid in ids:
            check = checks.get((pid, mid, "image_generation"))
            if check and check["result"]:
                evidence = 0
            elif check and not check["result"]:
                evidence = 4  # failed an image test: tried last, never excluded
            elif mid in declared:
                evidence = 1
            else:
                evidence = 2 if is_image_output_model(mid) else 3
            ranked.append(((evidence, BILLING_RANK.get(billing.get(mid, "unknown"), 1) if prefer_free else 0, rng.random()),
                           provider, mid))
    ranked.sort(key=lambda t: t[0])
    labels = {0: "a model that has produced images", 1: "a model you marked as image-capable",
              2: "an untested model (the app will try it and remember the result)",
              3: "an untested model", 4: "a model that failed an earlier image test"}
    return [(prov, mid, f"Auto-selected {prov.display_name} / {mid}: {labels[key[0]]}.") for key, prov, mid in ranked]


def _image_chat(request: ChatRequest, history: list[dict], context: str) -> dict[str, Any]:
    """Create an image. Manual: your model first. Auto-pick / Safe fallback: walk the candidates until one returns an image."""
    manual = not (request.auto or not request.provider_id)
    plan: list[tuple[Any, str, str]] = []
    if manual:
        plan.append(_image_target(request.provider_id, request.model_id, False, request.prefer_free))
    if (not manual) or request.safe_fallback:
        for cand in _image_candidates(request.prefer_free):
            if (cand[0].provider_id, cand[1]) not in {(c[0].provider_id, c[1]) for c in plan}:
                plan.append(cand)
        if not plan:
            raise HTTPException(status_code=400, detail=NO_IMAGE_MODEL_MESSAGE)
    limit = 6 if (request.safe_fallback or not manual) else 1
    texts = [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    log: list[str] = []
    text_only: dict[str, Any] | None = None
    for n, (provider, model_id, reason) in enumerate(plan[:limit]):
        try:
            verdict, payload = _try_image(provider, model_id, texts, context, reason)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            log.append(f"{provider.display_name} / {model_id}: {detail[:160]}")
            if limit == 1:
                raise
            continue
        except Exception as exc:  # noqa: BLE001
            log.append(redact(f"{provider.display_name} / {model_id}: {exc}")[:200])
            if limit == 1:
                raise HTTPException(status_code=502, detail=f"Model request failed: {redact(str(exc))}") from exc
            continue
        if verdict == "verified":
            if log:
                payload["fallback_reason"] = f"Earlier image attempts didn't work ({'; '.join(log)[:300]}); used {provider.display_name} / {model_id}."
            return payload
        log.append(f"{provider.display_name} / {model_id}: {'replied with text only' if verdict == 'text_only' else 'image request rejected' if verdict == 'rejected' else 'empty reply'}")
        if verdict == "text_only" and text_only is None:
            text_only = payload
        if limit == 1:
            if verdict == "rejected":
                raise HTTPException(status_code=400, detail=(
                    f"{provider.display_name} / {model_id} can't generate images: the provider rejected the image request. "
                    f"I've marked this model as not image-capable. Provider message: {payload[:200]}"))
            if verdict == "text_only":
                payload["capability_notice"] = "This model replied with text only, so it can't generate images."
                payload["answer"] = ("**This model can't generate images.** It replied with text only, so I've marked it as "
                                     "not image-capable. Here is what it said instead:\n\n" + payload["answer"])
            return payload
    if text_only is not None:
        text_only["capability_notice"] = "None of the models I tried returned an image; this is a text answer."
        return text_only
    raise HTTPException(status_code=400, detail="No model returned an image. Tried: " + " | ".join(log))


@app.post("/api/probe")
def probe_model(request: ProbeRequest) -> dict[str, Any]:
    """Model Lab: send small different requests to ONE model and record what it can really do.

    Only providers with a saved key can be probed. Each check is a real (tiny) request, so it can use a few tokens.
    """
    provider = next((p for p in providers if p.provider_id == request.provider_id), None)
    if provider is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider_id}'.")
    key = get_provider_key(provider.provider_id)
    if not key:
        raise HTTPException(status_code=400, detail=f"No API key saved for {provider.display_name}. Add one in Settings first.")
    kinds = [k for k in request.kinds if k in PROBE_KINDS] or ["text"]
    budget = check_budget(provider.provider_id, request.model_id, 600, 200, get_spend_config())
    if not budget.get("allowed", False):
        raise HTTPException(status_code=402, detail={"message": budget.get("reason", "Spending limit reached."), "budget": budget})
    pid, mid = provider.provider_id, request.model_id

    def record_for(capability: str, result: bool, evidence: str) -> None:
        registry.record_capability_check(pid, mid, capability, result, evidence)

    def send_with(_prepared: Prepared, messages: list) -> str:
        answer, usage, _images = _unpack(provider.chat(mid, messages, key, system_prompt=""))
        try:
            record_usage(pid, mid, "Model Lab probe", usage, None, None)
        except Exception:
            pass
        return answer

    sample = None
    if request.video_sample is not None:
        loaded = prepare_attachments([request.video_sample.model_dump()])
        sample = next((p.data for p in loaded.parts if p.kind == "video"), None)
    results = run_probe([k for k in kinds if k != "image_output"], send_with, record_for, video_sample=sample)

    if "image_output" in kinds:
        if pid not in registry.IMAGE_OUTPUT_ADAPTERS:
            results.append(ProbeResult("image_output", "skipped", "This app has no image-output adapter for this provider yet."))
        else:
            try:
                verdict, payload = _try_image(provider, mid, [{"role": "user", "content": PROBE_PROMPT}], "",
                                              f"Model Lab test of {provider.display_name} / {mid}.")
                status = {"verified": "pass", "text_only": "fail", "rejected": "fail"}.get(verdict, "inconclusive")
                detail = {"verified": "Returned an image.", "text_only": "Replied with text only.",
                          "rejected": "The provider rejected the image request.",
                          "inconclusive": "Empty reply (possibly a safety filter); nothing recorded."}[verdict]
                results.append(ProbeResult("image_output", status, detail, None, verdict != "inconclusive"))
            except HTTPException as exc:
                results.append(ProbeResult("image_output", "inconclusive", str(exc.detail)[:200]))
            except Exception as exc:  # noqa: BLE001
                results.append(ProbeResult("image_output", "inconclusive", str(exc)[:200]))
    order = {k: i for i, k in enumerate(PROBE_KINDS)}
    results.sort(key=lambda r: order.get(r.kind, 99))
    passed = [r.kind for r in results if r.status == "pass"]
    return {"provider_id": pid, "model_id": mid, "results": [r.as_dict() for r in results], "passed": passed}


@app.post("/api/models/test-image")
def test_image_output(request: ImageTestRequest) -> dict[str, Any]:
    """Send one small image request to find out whether this model can return an image, and remember it."""
    if request.provider_id not in get_provider_ids():
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider_id}'.")
    provider, model_id, reason = _image_target(request.provider_id, request.model_id, retest=True)
    verdict, payload = _try_image(provider, model_id, [{"role": "user", "content": PROBE_PROMPT}], "", reason)
    messages = {
        "verified": "This model returned an image.",
        "text_only": "This model replied with text only, so it can't generate images here.",
        "rejected": "The provider rejected the image request, so this model can't generate images.",
        "inconclusive": "The model returned nothing (possibly a safety filter). Nothing was recorded; try again.",
    }
    result: dict[str, Any] = {"status": verdict, "message": messages[verdict]}
    if verdict == "rejected":
        result["detail"] = payload[:200]
    return result
