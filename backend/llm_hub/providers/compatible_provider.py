
"""Adapter for OpenAI-compatible chat-completions endpoints."""

from typing import Any

from openai import OpenAI

from .base import ProviderAdapter, rough_message_estimate
from .parts import to_openai_content
from ..models import ModelInfo, UsageInfo


class CompatibleProvider(ProviderAdapter):
    provider_id = "compatible"
    display_name = "OpenAI-compatible API"

    def __init__(
        self,
        base_url: str,
        model_ids: list[str] | None = None,
        label: str = "OpenAI-compatible API",
        display_name: str | None = None,  # alias of label; router.py passes this name
    ):
        self.base_url = (base_url or "").strip()
        self.display_name = display_name or label
        self.model_ids = [
            model_id.strip()
            for model_id in (model_ids or [])
            if model_id and model_id.strip()
        ]

    def list_models(self) -> list[ModelInfo]:
        # These are configured IDs, not live-discovered model IDs.
        return [
            ModelInfo(
                self.display_name,
                model_id,
                model_id,
                None,
                ["text"],
                "configured model list",
            )
            for model_id in self.model_ids
        ]

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        api_key: str,
        system_prompt: str = "",
    ) -> tuple[str, UsageInfo]:
        if not self.base_url:
            raise ValueError(
                "Configure a base URL for this OpenAI-compatible endpoint."
            )

        if not api_key or not api_key.strip():
            raise ValueError("An API key is required for this endpoint.")

        if not model_id or not model_id.strip():
            raise ValueError("A model ID is required.")

        client = OpenAI(
            api_key=api_key.strip(),
            base_url=self.base_url,
            timeout=90,
        )

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

            # video_url is the convention several OpenAI-compatible gateways use; an endpoint that does not
            # understand it answers with an error, which the router treats as "try the next model".
            payload.append({"role": role, "content": to_openai_content(content, video_part=True)})

        if not payload:
            raise ValueError("At least one conversation message is required.")

        response = client.chat.completions.create(
            model=model_id.strip(),
            messages=payload,
        )

        answer = response.choices[0].message.content or ""
        metadata = response.usage

        input_tokens = getattr(metadata, "prompt_tokens", None) if metadata else None
        output_tokens = getattr(metadata, "completion_tokens", None) if metadata else None
        total_tokens = getattr(metadata, "total_tokens", None) if metadata else None

        usage = UsageInfo(
            input_tokens,
            output_tokens,
            total_tokens,
            "provider-reported" if total_tokens is not None else "estimated",
            None if total_tokens is not None
            else rough_message_estimate(messages, system_prompt),
        )

        return answer, usage
