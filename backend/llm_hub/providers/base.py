
"""Shared provider interface and conservative token estimation."""

from abc import ABC, abstractmethod
from typing import Any

from ..capabilities import UnsupportedModality  # noqa: F401  (adapters raise this for inputs they cannot carry)
from ..models import ModelInfo, UsageInfo


class ProviderAdapter(ABC):
    provider_id = "base"
    display_name = "Base provider"

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Return known models. Live availability is checked by the catalog."""
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        api_key: str,
        system_prompt: str = "",
    ) -> tuple[str, UsageInfo]:
        """Send a text conversation and return its response and usage."""
        raise NotImplementedError


def rough_token_estimate(text: str) -> int:
    """A rough character-based estimate, not an official tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def rough_message_estimate(
    messages: list[dict[str, Any]],
    system_prompt: str = "",
) -> int:
    total = rough_token_estimate(system_prompt)

    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):  # attachments: count the text, not the base64 bytes
            text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            text += " " * (1500 * 4 * sum(1 for b in content if isinstance(b, dict) and b.get("type") == "media"))
        else:
            text = str(content)

        total += rough_token_estimate(text) + 4

    return total
