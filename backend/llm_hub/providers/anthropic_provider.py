
"""Anthropic Claude text-generation adapter."""

from typing import Any

import anthropic

from .base import ProviderAdapter, rough_message_estimate
from .parts import to_anthropic_content
from ..models import ModelInfo, UsageInfo


class AnthropicProvider(ProviderAdapter):
    provider_id = "anthropic"
    display_name = "Anthropic Claude"

    def list_models(self) -> list[ModelInfo]:
        # Starter metadata, not a guarantee of access or modality support.
        return [
            ModelInfo(
                self.display_name,
                "claude-sonnet-4-20250514",
                "Claude Sonnet 4",
                None,
                ["text"],
                "starter catalog",
            ),
            ModelInfo(
                self.display_name,
                "claude-3-7-sonnet-latest",
                "Claude 3.7 Sonnet",
                None,
                ["text"],
                "starter catalog",
            ),
            ModelInfo(
                self.display_name,
                "claude-3-5-haiku-latest",
                "Claude 3.5 Haiku",
                None,
                ["text"],
                "starter catalog",
            ),
        ]

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        api_key: str,
        system_prompt: str = "",
    ) -> tuple[str, UsageInfo]:
        if not api_key or not api_key.strip():
            raise ValueError("An Anthropic API key is required.")

        client = anthropic.Anthropic(
            api_key=api_key.strip(),
            timeout=90,
        )

        payload = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in ("user", "assistant"):
                continue

            payload.append({"role": role, "content": to_anthropic_content(content)})

        if not payload:
            raise ValueError("At least one conversation message is required.")

        kwargs = {
            "model": model_id,
            "max_tokens": 2048,
            "messages": payload,
        }

        if system_prompt.strip():
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        answer = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

        metadata = getattr(response, "usage", None)
        input_tokens = getattr(metadata, "input_tokens", None)
        output_tokens = getattr(metadata, "output_tokens", None)

        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )

        usage = UsageInfo(
            input_tokens,
            output_tokens,
            total_tokens,
            "provider-reported" if total_tokens is not None else "estimated",
            None if total_tokens is not None
            else rough_message_estimate(messages, system_prompt),
        )

        return answer, usage
