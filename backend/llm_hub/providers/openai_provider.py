
"""OpenAI chat adapter."""

from typing import Any

from openai import OpenAI

from .base import ProviderAdapter, rough_message_estimate
from .parts import to_openai_content
from ..models import ModelInfo, UsageInfo


class OpenAIProvider(ProviderAdapter):
    provider_id = "openai"
    display_name = "OpenAI"

    def list_models(self) -> list[ModelInfo]:
        # Starter metadata only. The catalog should use the authenticated
        # models endpoint for the current account's discoverable model IDs.
        return [
            ModelInfo(
                self.display_name,
                "gpt-4.1-mini",
                "GPT-4.1 mini",
                None,
                ["text"],
                "starter catalog",
            ),
            ModelInfo(
                self.display_name,
                "gpt-4.1",
                "GPT-4.1",
                None,
                ["text"],
                "starter catalog",
            ),
            ModelInfo(
                self.display_name,
                "gpt-4o-mini",
                "GPT-4o mini",
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
            raise ValueError("An OpenAI API key is required.")

        client = OpenAI(api_key=api_key.strip(), timeout=90)

        payload: list[dict[str, Any]] = []

        if system_prompt.strip():
            payload.append({
                "role": "system",
                "content": system_prompt,
            })

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in ("user", "assistant"):
                continue

            payload.append({"role": role, "content": to_openai_content(content)})

        if not payload:
            raise ValueError("At least one non-empty conversation message is required.")

        response = client.chat.completions.create(
            model=model_id,
            messages=payload,
        )

        answer = response.choices[0].message.content or ""
        usage_data = response.usage

        input_tokens = getattr(usage_data, "prompt_tokens", None)
        output_tokens = getattr(usage_data, "completion_tokens", None)
        total_tokens = getattr(usage_data, "total_tokens", None)

        usage = UsageInfo(
            input_tokens,
            output_tokens,
            total_tokens,
            "provider-reported" if total_tokens is not None else "estimated",
            None if total_tokens is not None
            else rough_message_estimate(messages, system_prompt),
        )

        return answer, usage
