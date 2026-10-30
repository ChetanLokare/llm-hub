"""Capability knowledge for routing, learned from evidence - never from model names.

For every kind of input a request can carry (image, pdf, audio, video) a model is in one of three states:

  yes       a real request with that input succeeded on this model, the provider's own model list says it
            is supported, or you declared it
  no        the provider's own model list says it is NOT supported. The only status that removes a model.
  unlikely  a recent real request with that input was rejected. The model is tried LAST, never dropped:
            rejections can be transient or mis-worded, and they expire after NEGATIVE_TTL.
  unknown   nobody has said. The model is still a candidate: it is simply tried after the "yes" ones, and
            whatever happens is remembered.

Only the provider's own statement can take a model out of the running, so routing never rests on
hand-written assumptions about which vendor or model "can" do something, and one bad afternoon cannot lock
every model out.
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping

MEDIA_KINDS = ("image", "pdf", "audio", "video")
CHECK_PREFIX = "input:"
NEGATIVE_TTL = timedelta(days=3)

_ALIASES = {
    "text": "text", "image": "image", "images": "image", "vision": "image", "image_input": "image",
    "file": "pdf", "files": "pdf", "document": "pdf", "documents": "pdf", "pdf": "pdf", "pdf_input": "pdf",
    "audio": "audio", "audio_input": "audio", "video": "video", "video_input": "video",
}


def normalise_modalities(values) -> list[str]:
    out: list[str] = []
    for v in values or []:
        m = _ALIASES.get(str(v).strip().lower())
        if m and m not in out:
            out.append(m)
    return out


def declared_modalities(raw: Mapping | None) -> dict[str, bool] | None:
    """What a provider's own model-list entry says about accepted inputs, or None if it says nothing.

    * A list of modalities (OpenRouter ``architecture.input_modalities``) is authoritative: anything not
      listed is "no".
    * A capabilities mapping (Mistral ``vision``, Anthropic ``image_input`` / ``pdf_input``) only yields an
      answer for the keys it actually contains; a missing key stays unknown.
    """
    if not isinstance(raw, Mapping):
        return None
    listed = None
    arch = raw.get("architecture")
    if isinstance(arch, Mapping):
        if isinstance(arch.get("input_modalities"), list):
            listed = normalise_modalities(arch["input_modalities"])
        elif isinstance(arch.get("modality"), str) and "->" in arch["modality"]:
            listed = normalise_modalities(re.split(r"[+,\s]+", arch["modality"].split("->")[0]))
    for key in ("input_modalities", "supported_input_modalities"):
        if listed is None and isinstance(raw.get(key), list):
            listed = normalise_modalities(raw[key])
    if listed is not None:
        return {k: (k in listed) for k in MEDIA_KINDS}

    caps = raw.get("capabilities")
    if isinstance(caps, Mapping):
        found: dict[str, bool] = {}
        for key, kind in (("vision", "image"), ("image_input", "image"), ("pdf_input", "pdf"),
                          ("audio_input", "audio"), ("video_input", "video")):
            if key in caps:
                value = caps[key]
                value = value.get("supported") if isinstance(value, Mapping) else value
                if isinstance(value, bool):
                    found[kind] = found.get(kind, False) or value
        return found or None
    return None


def declared_text_output(raw: Mapping | None) -> bool | None:
    """False only when the provider explicitly says a model does not produce text (image generators etc.)."""
    if not isinstance(raw, Mapping):
        return None
    arch = raw.get("architecture")
    if isinstance(arch, Mapping) and isinstance(arch.get("output_modalities"), list):
        return "text" in [str(m).lower() for m in arch["output_modalities"]]
    kind = raw.get("type")
    if isinstance(kind, str) and kind.lower() in {"embedding", "image", "moderation", "rerank", "audio", "video"}:
        return False
    caps = raw.get("capabilities")
    if isinstance(caps, Mapping) and caps.get("completion_chat") is False:
        return False
    return None


def _parse_time(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def modality_status(kind: str, *, provider_id: str, model_id: str, declared: Mapping | None = None,
                    user_caps=(), checks: Mapping | None = None, now: datetime | None = None) -> tuple[str, str]:
    """Return (status, source) for one input kind. status is yes | no | unlikely | unknown."""
    now = now or datetime.now(timezone.utc)
    check = (checks or {}).get((provider_id, model_id, CHECK_PREFIX + kind))
    if check:
        if check["result"]:
            return "yes", "a real request succeeded"
        when = _parse_time(check.get("checked_at", ""))
        if when and now - when < NEGATIVE_TTL:
            return "unlikely", "a recent request was rejected"
    if isinstance(declared, Mapping) and kind in declared:
        return ("yes", "provider model list") if declared[kind] else ("no", "provider model list")
    if kind == "image" and "vision" in {str(c).lower() for c in user_caps or ()}:
        return "yes", "declared by you"
    return "unknown", "not tested yet"


def works_status(provider_id: str, model_id: str, checks: Mapping | None, now: datetime | None = None) -> str:
    """yes | unlikely | unknown for "does this model answer text requests at all", from Model Lab results.

    "unlikely" is a demotion (the model is tried last), never an exclusion, and it expires like other rejections.
    """
    now = now or datetime.now(timezone.utc)
    check = (checks or {}).get((provider_id, model_id, "works:text"))
    if not check:
        return "unknown"
    if check["result"]:
        return "yes"
    when = _parse_time(check.get("checked_at", ""))
    return "unlikely" if when and now - when < NEGATIVE_TTL else "unknown"


# ----------------------------------------------------------------------------- failures
class UnsupportedModality(ValueError):
    """Raised by an adapter that cannot carry a kind of input (e.g. video to a text-only API)."""

    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


_MODALITY_WORDS = re.compile(r"image|vision|multimodal|modalit|pdf|file|document|audio|video|mime|media|content[ -]type|input type")
_REJECT_WORDS = re.compile(r"not support|unsupported|invalid|does not|doesn't|cannot|can't|only text|unknown (variant|type)|"
                           r"expected|must be|no endpoints|not allowed|not enabled")


def _status_of(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    for value in (getattr(exc, "status_code", None), getattr(response, "status_code", None), getattr(exc, "code", None)):
        if isinstance(value, int):
            return value
    return None


def failure_kind(exc: BaseException) -> str:
    """rate_limit | auth | not_found | too_large | modality | server | bad_request | unknown"""
    if isinstance(exc, UnsupportedModality):
        return "modality"
    status = _status_of(exc)
    text = f"{type(exc).__name__} {exc}".lower()
    if status == 429 or "rate limit" in text or "rate_limit" in text or "resource_exhausted" in text or "quota" in text:
        return "rate_limit"
    if status in (401, 403) or "api key" in text or "unauthorized" in text or "permission denied" in text:
        return "auth"
    if status == 413 or "too large" in text or "context length" in text or "context_length" in text \
            or "maximum context" in text or "too many tokens" in text:
        return "too_large"
    if _MODALITY_WORDS.search(text) and _REJECT_WORDS.search(text) and (status is None or 400 <= status < 500):
        return "modality"
    if status == 404 or "not found" in text or "does not exist" in text:
        return "not_found"
    if (status is not None and status >= 500) or any(w in text for w in ("timeout", "timed out", "connection", "overloaded", "unavailable")):
        return "server"
    if status is not None and 400 <= status < 500:
        return "bad_request"
    return "unknown"


_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{10,}|gsk_[A-Za-z0-9]{10,}|Bearer\s+[A-Za-z0-9._\-]{10,})")


def redact(text: str) -> str:
    """Remove anything that looks like an API key or bearer token from text that may be shown to the user."""
    return _SECRET.sub("[redacted]", text)


def short_error(exc: BaseException, limit: int = 180) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    text = redact(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# ----------------------------------------------------------------------------- health
class Health:
    """Short-lived memory of what just failed, so a dead model or key is tried last instead of first.

    This only reorders; nothing is ever excluded for good.
    """
    COOLDOWN = {"rate_limit": 120, "server": 60, "not_found": 3600, "auth": 1800, "too_large": 0,
                "modality": 0, "bad_request": 0, "unknown": 30}

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._model: dict[tuple[str, str], float] = {}
        self._provider: dict[str, float] = {}

    def penalize(self, provider_id: str, model_id: str, kind: str) -> None:
        seconds = self.COOLDOWN.get(kind, 0)
        if seconds <= 0:
            return
        until = self._clock() + seconds
        with self._lock:
            if kind == "auth":
                self._provider[provider_id] = until
            else:
                self._model[(provider_id, model_id)] = until

    def is_cooling(self, provider_id: str, model_id: str) -> bool:
        now = self._clock()
        with self._lock:
            return self._provider.get(provider_id, 0) > now or self._model.get((provider_id, model_id), 0) > now

    def clear(self) -> None:
        with self._lock:
            self._model.clear()
            self._provider.clear()


HEALTH = Health()
RECENT_USE: dict[str, float] = {}  # provider_id -> monotonic time of last success; spreads load fairly
