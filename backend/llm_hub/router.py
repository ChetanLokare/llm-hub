"""Provider registry for LLM Hub.

This module registers provider adapters only. Live model discovery belongs
in catalog.py; this module does not invent model IDs or model capabilities.

Note:
- A provider being registered does not mean its API key is configured.
- A model appearing in a live catalog does not guarantee account-level access.
- Chat-compatible adapters do not automatically support image or video generation.
"""

from __future__ import annotations

import os
from typing import Any

from .models import ModelInfo
from .providers import (
    AnthropicProvider,
    CompatibleProvider,
    GeminiProvider,
    OpenAIProvider,
)


# Provider IDs must remain consistent with:
# - catalog.py endpoint/key mappings
# - main.py ENV_KEY_NAMES
# - API requests from the frontend
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai": {
        "label": "OpenAI",
        "url": "https://api.openai.com/v1",
    },
    "gemini": {
        "label": "Google Gemini",
        "url": "https://generativelanguage.googleapis.com",
    },
    "anthropic": {
        "label": "Anthropic",
        "url": "https://api.anthropic.com",
    },
    "openrouter": {
        "label": "OpenRouter",
        "url": "https://openrouter.ai/api/v1",
    },
    "groq": {
        "label": "Groq",
        "url": "https://api.groq.com/openai/v1",
    },
    "deepseek": {
        "label": "DeepSeek",
        "url": "https://api.deepseek.com/v1",
    },
    "mistral": {
        "label": "Mistral AI",
        "url": "https://api.mistral.ai/v1",
    },
    "together": {
        "label": "Together AI",
        "url": "https://api.together.xyz/v1",
    },
}


class RegisteredCompatibleProvider(CompatibleProvider):
    """OpenAI-compatible provider registration without a fabricated model list.

    The model list is intentionally empty until live discovery or the
    "Add model" option supplies IDs. CompatibleProvider is still a chat
    adapter; it does not imply support for media-generation endpoints.
    """

    def __init__(
        self,
        provider_id: str,
        display_name: str,
        base_url: str,
        model_ids: list[str] | None = None,
    ) -> None:
        # Positional arguments work with every version of CompatibleProvider:
        # its third parameter is the label, and newer versions also accept
        # display_name=. Keyword arguments broke on the older signature.
        super().__init__(base_url, list(model_ids or []), display_name)
        self.provider_id = provider_id
        self.display_name = display_name
        self.base_url = base_url
        # CompatibleProvider substitutes a placeholder model ID when it is given
        # an empty list. Keep the list truly empty so no model is invented.
        self.model_ids = list(model_ids or [])

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(self.display_name, model_id, model_id, None, ["text"], "configured model")
            for model_id in self.model_ids
        ]


def _split_model_ids(value: str | None) -> list[str]:
    """Parse a comma-separated model list and remove duplicates."""
    if not value:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for raw_model_id in value.split(","):
        model_id = raw_model_id.strip()
        if model_id and model_id not in seen:
            result.append(model_id)
            seen.add(model_id)

    return result


def _provider_id(provider: Any) -> str:
    """Return a stable provider ID, deriving one only when necessary."""
    existing = getattr(provider, "provider_id", None)
    if isinstance(existing, str) and existing.strip():
        return existing.strip().lower()

    display_name = str(getattr(provider, "display_name", "")).strip().lower()
    aliases = {
        "openai": "openai",
        "google gemini": "gemini",
        "gemini": "gemini",
        "anthropic": "anthropic",
        "claude": "anthropic",
    }

    return aliases.get(display_name, display_name.replace(" ", "_"))


def _assign_provider_id(provider: Any, provider_id: str) -> Any:
    """Ensure adapters expose the IDs expected by the rest of the backend."""
    provider.provider_id = provider_id
    return provider


def build_providers(
    base_url: str = "",
    models: list[str] | None = None,
    label: str = "OpenAI-compatible API",
) -> list[Any]:
    """Build the supported provider adapters.

    Built-in adapters:
        OpenAI, Gemini, Anthropic, OpenRouter, Groq, DeepSeek,
        Mistral AI, and Together AI.

    Optional custom endpoint environment variables:
        LLMHUB_COMPATIBLE_BASE_URL
        LLMHUB_COMPATIBLE_MODELS
        LLMHUB_COMPATIBLE_LABEL

    No hardcoded model catalog is created here. The catalog module is
    responsible for fetching model IDs from each provider's API.
    """
    providers: list[Any] = [
        _assign_provider_id(OpenAIProvider(), "openai"),
        _assign_provider_id(GeminiProvider(), "gemini"),
        _assign_provider_id(AnthropicProvider(), "anthropic"),
    ]

    # Register OpenAI-compatible provider APIs. No model IDs or capabilities
    # are assumed here; catalog.py performs model discovery separately.
    for provider_id in (
        "openrouter",
        "groq",
        "deepseek",
        "mistral",
        "together",
    ):
        spec = PROVIDER_DEFAULTS[provider_id]
        providers.append(
            RegisteredCompatibleProvider(
                provider_id=provider_id,
                display_name=spec["label"],
                base_url=spec["url"],
            )
        )

    env_base_url = os.environ.get(
        "LLMHUB_COMPATIBLE_BASE_URL", ""
    ).strip()
    env_models = _split_model_ids(
        os.environ.get("LLMHUB_COMPATIBLE_MODELS", "")
    )
    env_label = os.environ.get(
        "LLMHUB_COMPATIBLE_LABEL", ""
    ).strip()

    custom_base_url = (base_url or env_base_url).strip()
    custom_models = (
        list(dict.fromkeys(m.strip() for m in models if m.strip()))
        if models is not None
        else env_models
    )
    custom_label = (label or env_label or "OpenAI-compatible API").strip()

    if custom_base_url:
        custom_provider = CompatibleProvider(
            custom_base_url,
            custom_models or [],
            custom_label,
        )
        _assign_provider_id(custom_provider, "compatible")
        providers.append(custom_provider)

    return providers


def get_provider(providers: list[Any], provider_id: str) -> Any:
    """Return a provider by ID, raising a clear error when it is unknown."""
    requested_id = (provider_id or "").strip().lower()

    for provider in providers:
        if _provider_id(provider) == requested_id:
            return provider

    available = ", ".join(
        _provider_id(provider) for provider in providers
    )
    raise LookupError(
        f"Unknown provider '{provider_id}'. Available providers: {available}"
    )
