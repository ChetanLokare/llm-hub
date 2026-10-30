
"""
Dynamic model catalog for LLM Hub.

Public interface:
    get_catalog(providers, get_provider_key, refresh=False)

Supports live model discovery for common LLM providers, in-memory caching,
provider-defined models, and clearly labelled fallback models.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .capabilities import declared_modalities, declared_text_output


CACHE_TTL_SECONDS = max(0, int(os.getenv("LLM_HUB_CATALOG_TTL", "900")))
REQUEST_TIMEOUT_SECONDS = max(1, int(os.getenv("LLM_HUB_CATALOG_TIMEOUT", "8")))

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


# These are fallback choices, not guarantees of current account access.
FALLBACK_MODELS: dict[str, list[dict[str, Any]]] = {
    "openai": [
        {
            "model_id": "gpt-4o-mini",
            "display_name": "GPT-4o mini",
            "capabilities": ["chat", "text"],
            "context_window": 128000,
        },
        {
            "model_id": "gpt-4o",
            "display_name": "GPT-4o",
            "capabilities": ["chat", "text"],
            "context_window": 128000,
        },
    ],
    "gemini": [
        {
            "model_id": "gemini-2.5-flash",
            "display_name": "Gemini 2.5 Flash",
            "capabilities": ["chat", "text"],
            "context_window": 1048576,
        },
        {
            "model_id": "gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "capabilities": ["chat", "text"],
            "context_window": 1048576,
        },
    ],
    "anthropic": [
        {
            "model_id": "claude-3-5-haiku-latest",
            "display_name": "Claude Haiku",
            "capabilities": ["chat", "text"],
            "context_window": 200000,
        },
        {
            "model_id": "claude-3-7-sonnet-latest",
            "display_name": "Claude Sonnet",
            "capabilities": ["chat", "text"],
            "context_window": 200000,
        },
    ],
    "openrouter": [
        {
            "model_id": "openai/gpt-4o-mini",
            "display_name": "OpenAI GPT-4o mini",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
        {
            "model_id": "anthropic/claude-3.5-sonnet",
            "display_name": "Anthropic Claude Sonnet",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
    ],
    "groq": [
        {
            "model_id": "llama-3.3-70b-versatile",
            "display_name": "Llama 3.3 70B",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
        {
            "model_id": "llama-3.1-8b-instant",
            "display_name": "Llama 3.1 8B Instant",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
    ],
    "deepseek": [
        {
            "model_id": "deepseek-chat",
            "display_name": "DeepSeek Chat",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
        {
            "model_id": "deepseek-reasoner",
            "display_name": "DeepSeek Reasoner",
            "capabilities": ["chat", "text", "reasoning"],
            "context_window": None,
        },
    ],
    "mistral": [
        {
            "model_id": "mistral-small-latest",
            "display_name": "Mistral Small",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
        {
            "model_id": "mistral-large-latest",
            "display_name": "Mistral Large",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
    ],
    "together": [
        {
            "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "display_name": "Llama 3.1 8B Instruct Turbo",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
        {
            "model_id": "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo",
            "display_name": "Llama 3.3 70B Instruct Turbo",
            "capabilities": ["chat", "text"],
            "context_window": None,
        },
    ],
    "compatible": [
        {
            "model_id": "local-model",
            "display_name": "Compatible API model",
            "capabilities": ["chat", "text"],
            "context_window": None,
        }
    ],
    "ollama": [
        {
            "model_id": "llama3.2",
            "display_name": "Llama 3.2 (Ollama)",
            "capabilities": ["chat", "text"],
            "context_window": None,
        }
    ],
}


# Provider model-list endpoints.
ENDPOINTS: dict[str, dict[str, str]] = {
    "openai": {
        "url": "https://api.openai.com/v1/models",
        "auth": "bearer",
        "format": "data",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/models",
        "auth": "bearer",
        "format": "data",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/models",
        "auth": "bearer",
        "format": "data",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/models",
        "auth": "bearer",
        "format": "data",
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/models",
        "auth": "bearer",
        "format": "data",
    },
    "together": {
        "url": "https://api.together.xyz/v1/models",
        "auth": "bearer",
        "format": "data",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/models",
        "auth": "anthropic",
        "format": "data",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "auth": "google",
        "format": "models",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _provider_id(provider: Any) -> str:
    if isinstance(provider, Mapping):
        value = (
            provider.get("provider_id")
            or provider.get("id")
            or provider.get("name")
        )
    else:
        value = (
            getattr(provider, "provider_id", None)
            or getattr(provider, "id", None)
            or getattr(provider, "name", None)
        )

    return _text(value).lower()


def _provider_name(provider: Any, provider_id: str) -> str:
    if isinstance(provider, Mapping):
        value = (
            provider.get("display_name")
            or provider.get("name")
        )
    else:
        value = (
            getattr(provider, "display_name", None)
            or getattr(provider, "name", None)
        )

    return _text(value) or provider_id.replace("_", " ").title()


def _providers_list(providers: Any) -> list[Any]:
    if providers is None:
        return []

    if isinstance(providers, Mapping):
        result = []

        for key, value in providers.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("provider_id", key)
                result.append(item)
            else:
                result.append(value)

        return result

    if isinstance(providers, (list, tuple, set)):
        return list(providers)

    try:
        return list(providers)
    except TypeError:
        return [providers]


def _get_key(
    get_provider_key: Callable[[str], Any],
    provider_id: str,
) -> str | None:
    try:
        key = get_provider_key(provider_id)
    except Exception:
        return None

    if isinstance(key, str) and key.strip():
        return key.strip()

    return None


def _http_get_json(
    url: str,
    headers: dict[str, str],
) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLM-Hub/1.0",
            **headers,
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(encoding))

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Provider model-list request returned HTTP {exc.code}"
        ) from None

    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError(
            "Provider model-list endpoint could not be reached or timed out"
        ) from None

    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError(
            "Provider returned an invalid JSON response"
        ) from None


def _normalise_model(
    item: Any,
    provider_id: str,
) -> dict[str, Any] | None:
    if isinstance(item, str):
        model_id = item.strip()
        raw: Mapping[str, Any] = {}
    elif isinstance(item, Mapping):
        raw = item
        model_id = _text(
            raw.get("model_id")
            or raw.get("id")
            or raw.get("name")
        )
    else:
        return None

    if not model_id:
        return None

    if provider_id == "gemini" and model_id.startswith("models/"):
        model_id = model_id.removeprefix("models/")

    lower_id = model_id.lower()

    # Avoid showing models that are obviously not general chat models.
    if any(
        term in lower_id
        for term in (
            "embedding",
            "moderation",
            "whisper",
            "text-to-speech",
            "dall-e",
            "imagen",
        )
    ):
        return None

    # Provider metadata only. A model's NAME is never used to claim it can see images, read PDFs, etc.
    if declared_text_output(raw) is False:
        return None  # the provider itself says this model does not produce text (image generator, embedder...)

    capabilities = raw.get("capabilities", ["chat", "text"])

    if not isinstance(capabilities, list):
        capabilities = ["chat", "text"]

    if provider_id == "gemini":
        methods = raw.get("supportedGenerationMethods")
        if isinstance(methods, list) and not any("generateContent" in str(method) for method in methods):
            return None

    modality_support = declared_modalities(raw)
    if modality_support and modality_support.get("image") and "vision" not in capabilities:
        capabilities = [*capabilities, "vision"]

    context = (
        raw.get("context_window")
        or raw.get("context_length")
        or raw.get("input_token_limit")
        or raw.get("max_input_tokens")
    )

    try:
        context = int(context) if context is not None else None
    except (TypeError, ValueError):
        context = None

    display_name = _text(raw.get("display_name"))

    if not display_name:
        display_name = model_id.replace("/", " / ").replace("-", " ")

    return {
        "model_id": model_id,
        "display_name": display_name,
        "capabilities": [str(capability) for capability in capabilities],
        "context_window": context,
        # What the provider says the model accepts as input; None when its list says nothing.
        "modality_support": modality_support,
        # Provider-published per-token pricing (OpenRouter publishes it). Used only to classify free vs paid.
        "pricing": dict(raw["pricing"]) if isinstance(raw.get("pricing"), Mapping) else None,
    }


def _fetch_live_models(
    provider_id: str,
    api_key: str,
) -> list[dict[str, Any]]:
    spec = ENDPOINTS.get(provider_id)

    if not spec:
        raise RuntimeError("Live discovery is not configured for this provider")

    headers: dict[str, str] = {}

    if spec["auth"] == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"

    elif spec["auth"] == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"

    elif spec["auth"] == "google":
        headers["x-goog-api-key"] = api_key

    payload = _http_get_json(spec["url"], headers)

    if not isinstance(payload, Mapping):
        raise RuntimeError("Provider returned an unexpected model-list format")

    items = payload.get(spec["format"], [])

    if not isinstance(items, list):
        raise RuntimeError("Provider returned an unexpected model-list format")

    models = []

    for item in items:
        model = _normalise_model(item, provider_id)

        if model:
            models.append(model)

    if not models:
        raise RuntimeError("Provider returned no compatible chat models")

    # Keep the result stable and avoid duplicate IDs.
    unique: dict[str, dict[str, Any]] = {}

    for model in models:
        unique[model["model_id"]] = model

    return sorted(
        unique.values(),
        key=lambda model: model["model_id"].lower(),
    )


def _fallback_models(
    provider_id: str,
    provider: Any,
) -> list[dict[str, Any]]:
    if isinstance(provider, Mapping):
        supplied = provider.get("models")
    else:
        supplied = getattr(provider, "models", None)

    result: list[dict[str, Any]] = []

    if isinstance(supplied, (list, tuple)):
        for item in supplied:
            model = _normalise_model(item, provider_id)

            if model:
                result.append(model)

    if not result:
        result = [
            dict(model)
            for model in FALLBACK_MODELS.get(provider_id, [])
        ]

    # Remove duplicates from provider-defined lists too.
    unique = {model["model_id"]: model for model in result}

    return list(unique.values())


def _build_entry(
    provider: Any,
    get_provider_key: Callable[[str], Any],
    refresh: bool,
) -> dict[str, Any]:
    provider_id = _provider_id(provider)

    if not provider_id:
        return {}

    display_name = _provider_name(provider, provider_id)
    now = time.time()

    with _CACHE_LOCK:
        cached = _CACHE.get(provider_id)

        if (
            not refresh
            and cached
            and now - cached["cached_at"] < CACHE_TTL_SECONDS
        ):
            entry = dict(cached["entry"])
            entry["display_name"] = display_name
            return entry

    api_key = _get_key(get_provider_key, provider_id)
    models: list[dict[str, Any]] = []
    source = "fallback"
    error_message = ""

    if api_key and provider_id not in ENDPOINTS:
        base = (provider.get("base_url") if isinstance(provider, Mapping) else getattr(provider, "base_url", "")) or ""
        if base.strip():
            ENDPOINTS[provider_id] = {"url": base.strip().rstrip("/") + "/models", "auth": "bearer", "format": "data"}

    if api_key and provider_id in ENDPOINTS:
        try:
            models = _fetch_live_models(provider_id, api_key)
            source = "live"
        except Exception as exc:
            # Do not expose API keys, HTTP bodies, or raw provider diagnostics.
            error_message = str(exc)

    if not models:
        models = _fallback_models(provider_id, provider)

    if source == "live":
        message = f"Live model list retrieved from {display_name}."
    elif api_key and provider_id in ENDPOINTS:
        message = (
            "Live discovery failed; showing fallback models. "
            + (error_message or "Check provider credentials and connectivity.")
        )
    elif not api_key:
        message = (
            "Showing fallback models. Configure an API key to attempt "
            "live model discovery."
        )
    else:
        message = (
            "Showing provider-defined or fallback models. "
            "Live discovery is not configured for this provider."
        )

    entry = {
        "provider_id": provider_id,
        "display_name": display_name,
        "models": models,
        "catalog_source": source,
        "catalog_message": message,
        "last_checked": _now(),
    }

    with _CACHE_LOCK:
        _CACHE[provider_id] = {
            "cached_at": now,
            "entry": entry,
        }

    return entry


def get_catalog(
    providers: Any,
    get_provider_key: Callable[[str], Any],
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Return the provider catalog expected by the FastAPI backend and React UI.

    Args:
        providers:
            Iterable of provider objects, a mapping of provider IDs to
            provider objects, or a single provider object.
        get_provider_key:
            Function that accepts a provider ID and returns its API key,
            or None when no key is configured.
        refresh:
            If True, bypass the cache and retry live discovery.

    Returns:
        A list of provider dictionaries with model metadata and catalog status.
    """
    if not callable(get_provider_key):
        raise TypeError("get_provider_key must be callable")

    catalog: list[dict[str, Any]] = []

    for provider in _providers_list(providers):
        try:
            entry = _build_entry(
                provider,
                get_provider_key,
                refresh,
            )

            if entry:
                catalog.append(entry)

        except Exception:
            # A single malformed provider should not break the whole endpoint.
            provider_id = _provider_id(provider)

            if provider_id:
                catalog.append({
                    "provider_id": provider_id,
                    "display_name": _provider_name(provider, provider_id),
                    "models": _fallback_models(provider_id, provider),
                    "catalog_source": "fallback",
                    "catalog_message": (
                        "Catalog discovery failed; showing fallback models."
                    ),
                    "last_checked": _now(),
                })

    return catalog


def clear_catalog_cache() -> None:
    """Clear cached catalog results."""
    with _CACHE_LOCK:
        _CACHE.clear()
