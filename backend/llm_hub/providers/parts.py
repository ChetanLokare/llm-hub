"""Convert LLM Hub's neutral message content into each provider's wire format (no SDK imports).

Neutral content is either a str or a list of blocks:
  {"type": "text", "text": ...}
  {"type": "media", "kind": image|pdf|audio|video, "mime_type", "name", "data": base64}

A converter raises UnsupportedModality when the API genuinely has no way to carry a kind of input. The router
treats that as "try another model", never as a reason to stop.
"""
from __future__ import annotations

from typing import Any

from ..capabilities import UnsupportedModality

OPENAI_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
ANTHROPIC_IMAGE_TYPES = OPENAI_IMAGE_TYPES
GEMINI_INLINE_LIMIT = 18 * 1024 * 1024  # inline request bodies are capped near 20 MB


def blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    raise ValueError("Message content must be text or a list of content blocks.")


def to_openai_content(content: Any, *, video_part: bool = False) -> str | list[dict]:
    """OpenAI chat-completions format; also used by OpenAI-compatible APIs (OpenRouter, Groq, Mistral...)."""
    if isinstance(content, str):
        return content
    out: list[dict] = []
    for b in blocks(content):
        if b.get("type") == "text":
            out.append({"type": "text", "text": b["text"]})
            continue
        kind, mime, data, name = b["kind"], b["mime_type"], b["data"], b.get("name") or "file"
        if kind == "image":
            if mime not in OPENAI_IMAGE_TYPES:
                raise UnsupportedModality(f"Image type {mime} is not supported by this API.", "image")
            out.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        elif kind == "pdf":
            out.append({"type": "file", "file": {"filename": name.split("/")[-1] or "document.pdf",
                                                 "file_data": f"data:application/pdf;base64,{data}"}})
        elif kind == "audio":
            fmt = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}.get(mime)
            if not fmt:
                raise UnsupportedModality(f"Audio type {mime} is not supported by this API (wav and mp3 only).", "audio")
            out.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
        elif kind == "video":
            if not video_part:
                raise UnsupportedModality("This API has no video input.", "video")
            out.append({"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}})
        else:
            raise UnsupportedModality(f"Unsupported attachment kind {kind}.", kind)
    return out


def to_anthropic_content(content: Any) -> str | list[dict]:
    if isinstance(content, str):
        return content
    out: list[dict] = []
    for b in blocks(content):
        if b.get("type") == "text":
            out.append({"type": "text", "text": b["text"]})
            continue
        kind, mime, data = b["kind"], b["mime_type"], b["data"]
        if kind == "image":
            if mime not in ANTHROPIC_IMAGE_TYPES:
                raise UnsupportedModality(f"Image type {mime} is not supported by this API.", "image")
            out.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
        elif kind == "pdf":
            out.append({"type": "document", "title": (b.get("name") or "document").split("/")[-1][:200],
                        "source": {"type": "base64", "media_type": "application/pdf", "data": data}})
        else:
            raise UnsupportedModality(f"This API has no {kind} input.", kind)
    return out


def to_gemini_items(content: Any) -> list[tuple]:
    """Return [("text", str) | ("bytes", mime, raw_bytes)] for the Gemini adapter to wrap in SDK types."""
    import base64
    items: list[tuple] = []
    total = 0
    for b in blocks(content):
        if b.get("type") == "text":
            items.append(("text", b["text"]))
            continue
        raw = base64.b64decode(b["data"])
        total += len(raw)
        if total > GEMINI_INLINE_LIMIT:
            raise UnsupportedModality("Attachments are too large to send inline to this API.", b.get("kind"))
        items.append(("bytes", b["mime_type"], raw))
    return items
