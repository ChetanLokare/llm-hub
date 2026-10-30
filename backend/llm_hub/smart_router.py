"""Conservative routing, context packing and safe fallback for LLM Hub.

Fallback policy deliberately retries only explicit HTTP 429 rate-limit responses.
Timeouts, connection resets, server errors and generic errors are ambiguous: the
provider may already have processed/billed the request, so they are never retried.
"""
from dataclasses import dataclass, field
from typing import Callable

def _content_text(content):
    """Text of a message; attachment blocks count as a flat allowance instead of their base64 bytes."""
    if isinstance(content, list):
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        return text + " " * (6000 * sum(1 for b in content if isinstance(b, dict) and b.get("type") == "media"))
    return str(content or "")


def rough_message_estimate(messages, system_prompt=""):
    # Deliberately rough fallback; this is not an official provider tokenizer.
    text = system_prompt + "".join(_content_text(m.get("content", "")) for m in messages)
    return max(1, (len(text) + 3) // 4) + 4 * len(messages) if text or messages else 0

@dataclass
class RouteResult:
    answer: str
    usage: object
    provider: object
    model_id: str
    context_note: str
    fallback_reason: str = ""
    images: list = field(default_factory=list)  # [{"mime_type", "data": base64}] from image-capable models


def _unpack(outcome):
    """Adapters return (text, usage) or (text, usage, images)."""
    if len(outcome) == 3:
        return outcome
    answer, usage = outcome
    return answer, usage, []


def exception_status(exc):
    response = getattr(exc, "response", None)
    return getattr(exc, "status_code", None) or getattr(response, "status_code", None)


def is_explicit_rate_limit(exc):
    return exception_status(exc) == 429


def _summary_line(role, content, limit=360):
    text = " ".join(_content_text(content).split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " … [older message abbreviated]"
    return f"{role.upper()}: {text}"


def pack_context(messages, context_window=None, reserve_output=2048, system_prompt=""):
    """Preserve newest turns verbatim and compress old turns locally, without API cost.

    Uses a conservative character/4 estimate, not a provider tokenizer. If a model's
    context size is unknown, keeps the latest 40 messages and labels the estimate.
    """
    if not messages:
        return [], "No prior context."
    budget = max(1000, int(context_window * 0.72) - reserve_output) if context_window else 24000
    recent = []
    used = rough_message_estimate([], system_prompt)
    for msg in reversed(messages):
        cost = rough_message_estimate([msg])
        if used + cost > budget and recent:
            break
        recent.append(dict(msg)); used += cost
    recent.reverse()
    omitted = messages[:len(messages)-len(recent)]
    if omitted:
        summary = "Earlier conversation context (locally compressed; original transcript remains saved):\n" + "\n".join(
            _summary_line(m.get("role", "unknown"), m.get("content", "")) for m in omitted[-30:])
        summary_msg = {"role": "user", "content": summary}
        packed = [summary_msg] + recent
        note = f"Preserved {len(messages)} messages; sent {len(recent)} recent messages plus locally compressed summary of {len(omitted)} older messages. Token fit is approximate."
    else:
        packed = recent
        note = f"Sent {len(recent)} messages; token fit is approximate."
    # Unknown context size: cap message count to keep payload bounded.
    if not context_window and len(packed) > 41:
        packed = packed[-41:]
        note += " Unknown model context window; kept at most 41 payload messages."
    return packed, note


class SmartRouter:
    def __init__(self, providers, key_lookup: Callable[[str], str | None], fallback_enabled=False):
        self.providers = providers
        self.key_lookup = key_lookup
        self.fallback_enabled = fallback_enabled

    def run(self, primary_provider, model_id, messages, system_prompt="", want_images=False):
        model = next((m for m in primary_provider.list_models() if m.model_id == model_id), None)
        packed, note = pack_context(messages, model.context_window if model else None, system_prompt=system_prompt)
        # Only adapters that can return images are asked to; other adapters keep their original signature.
        extra = {"want_images": True} if want_images else {}
        try:
            key = self.key_lookup(primary_provider.provider_id)
            if not key:
                raise ValueError(f"No API key saved for {primary_provider.display_name}.")
            answer, usage, images = _unpack(primary_provider.chat(model_id, packed, key, system_prompt=system_prompt, **extra))
            return RouteResult(answer, usage, primary_provider, model_id, note, images=images)
        except Exception as exc:
            if want_images or not self.fallback_enabled or not is_explicit_rate_limit(exc):
                raise
            # Only explicit 429s are considered safe to fail over. Never retry timeouts,
            # network errors, 5xx, authentication, or generic exceptions because billing
            # may be ambiguous. Do not retry the same provider/model.
            for provider in self.providers:
                if provider.provider_id == primary_provider.provider_id:
                    continue
                key = self.key_lookup(provider.provider_id)
                if not key:
                    continue
                for candidate in provider.list_models():
                    try:
                        answer, usage, images = _unpack(provider.chat(candidate.model_id, packed, key, system_prompt=system_prompt))
                        reason = f"Primary provider returned HTTP 429; switched to {provider.display_name} / {candidate.model_id}."
                        return RouteResult(answer, usage, provider, candidate.model_id, note, reason, images=images)
                    except Exception as fallback_exc:
                        # A failed fallback is not retried again unless it is a 429; even
                        # then this MVP avoids cascaded retries to minimize surprise costs.
                        if is_explicit_rate_limit(fallback_exc):
                            continue
                        raise fallback_exc
            raise exc
