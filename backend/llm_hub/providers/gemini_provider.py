
"""Google Gemini text-generation adapter."""

from typing import Any

from google import genai
from google.genai import types

from .base import ProviderAdapter, rough_message_estimate
from .media import is_image_output_model, parts_to_text_and_images
from .parts import to_gemini_items
from ..models import ModelInfo, UsageInfo


class GeminiProvider(ProviderAdapter):
    provider_id = "gemini"
    display_name = "Google Gemini"

    def list_models(self) -> list[ModelInfo]:
        # Do not infer vision or generation support from a model name.
        # Use live catalog data for discovery and separately verified metadata
        # for capabilities.
        return [
            ModelInfo(
                self.display_name,
                "gemini-2.5-flash",
                "Gemini 2.5 Flash",
                None,
                ["text"],
                "starter catalog",
            ),
            ModelInfo(
                self.display_name,
                "gemini-2.5-pro",
                "Gemini 2.5 Pro",
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
        want_images: bool = False,
    ) -> tuple:  # (text, usage, images)
        if not api_key or not api_key.strip():
            raise ValueError("A Gemini API key is required.")

        client = genai.Client(api_key=api_key.strip())
        contents = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in ("user", "assistant"):
                continue

            parts = [
                types.Part.from_text(text=item[1]) if item[0] == "text"
                else types.Part.from_bytes(data=item[2], mime_type=item[1])
                for item in to_gemini_items(content)
            ]
            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=parts,
                )
            )

        if not contents:
            raise ValueError("At least one conversation message is required.")

        # Image-capable Gemini models must be asked for image output explicitly.
        options = {"response_modalities": ["TEXT", "IMAGE"]} if (want_images or is_image_output_model(model_id)) else {}
        config = types.GenerateContentConfig(
            system_instruction=system_prompt.strip() or None,
            **options,
        )

        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=config,
        )

        parts = []
        for candidate in (getattr(response, "candidates", None) or [])[:1]:
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        answer, images = parts_to_text_and_images(parts)
        if not answer and not images:
            try:
                answer = response.text or ""
            except Exception:
                answer = ""
        metadata = getattr(response, "usage_metadata", None)

        input_tokens = getattr(metadata, "prompt_token_count", None)
        output_tokens = getattr(metadata, "candidates_token_count", None)
        total_tokens = getattr(metadata, "total_token_count", None)

        usage = UsageInfo(
            input_tokens,
            output_tokens,
            total_tokens,
            "provider-reported" if total_tokens is not None else "estimated",
            None if total_tokens is not None
            else rough_message_estimate(messages, system_prompt),
        )

        return answer, usage, images
