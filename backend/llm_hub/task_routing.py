"""Task-aware model ranking and transparent local spend estimates.

NOTE: the name-based tier ranking below (rank_models/choose_model/model_profile) is LEGACY. Chat requests
are now routed by auto_router.py, which uses provider-reported capabilities and observed results instead of
model names. This module is still used for manual selection, media tasks and cost estimates.

Pricing is user-configured because provider prices change and compatible endpoints
may have custom rates. No quota values are inferred from token usage.
"""
import math
import re
from dataclasses import dataclass

TASKS = {
    "General chat": {"text"},
    "Coding": {"coding", "analysis", "text"},
    "Research": {"analysis", "text"},
    "Writing": {"text", "fast"},
    "Document analysis": {"analysis", "vision", "text"},
    "Image generation": {"image_generation"},
    "Video generation": {"video_generation"},
}

# ---------------------------------------------------------------------------
# Model profiling: infer tier / reasoning / coding / vision from the model id.
# This is a name-and-metadata heuristic, not a benchmark. It works on ids it has
# never seen (live-discovered models), and can later be replaced by measured
# per-task scores kept in a shared catalog.
# ---------------------------------------------------------------------------
TIER_LABELS = {0: "tiny", 1: "light", 2: "strong", 3: "flagship"}

# target tier, and weights for reasoning / coding / long-context / vision
TASK_PROFILE = {
    "General chat":      dict(tier=1, reasoning=-1.5, coding=0.0, context=0.0, vision=0.0),
    "Writing":           dict(tier=2, reasoning=-1.0, coding=0.0, context=0.3, vision=0.0),
    "Coding":            dict(tier=3, reasoning=1.5,  coding=3.0, context=0.5, vision=0.0),
    "Research":          dict(tier=3, reasoning=2.5,  coding=0.0, context=1.5, vision=0.0),
    "Document analysis": dict(tier=2, reasoning=0.0,  coding=0.0, context=2.5, vision=1.5),
}

_PARAMS = re.compile(r"(?<![a-z0-9.])(?:(\d+)x)?(\d+(?:\.\d+)?)b(?![a-z0-9])")
_VERSION = re.compile(r"(\d+)(?:[.-](\d))?(?!\d)")


def _params_b(mid):
    m = _PARAMS.search(mid)
    if not m:
        return None
    return float(m.group(2)) * (int(m.group(1)) if m.group(1) else 1)


def _version(mid):
    m = _VERSION.search(_PARAMS.sub("", mid))
    if not m:
        return 0.0
    return float(m.group(1)) + (int(m.group(2)) / 10 if m.group(2) else 0)


def model_profile(model):
    """Return a dict describing a model, inferred from its id and declared capabilities."""
    mid = model.model_id.lower()
    caps = {str(c).lower() for c in model.capabilities}
    params = _params_b(mid)

    if re.search(r"nano|lite|tiny|instant|ministral", mid) or (params is not None and params <= 4):
        tier = 0
    elif re.search(r"(?<![a-z])mini|flash|haiku|small|gemma", mid) or (params is not None and 5 <= params <= 36):
        tier = 1
    elif re.search(r"opus|ultra|[-/]pro\b|reasoner|thinking|(^|[/-])r1\b|(^|[/-])o[134]([-/]|$)|gpt-5", mid) \
            or (params is not None and params >= 200):
        tier = 3
    elif re.search(r"sonnet|large|maverick|medium|gpt-4|deepseek-chat|codestral", mid) \
            or (params is not None and params >= 37):
        tier = 2
    else:
        tier = 1.5

    reasoning = bool(re.search(r"reasoner|thinking|qwq|(^|[/-])r1\b|(^|[/-])o[134]([-/]|$)|gpt-oss|qwen3|gpt-5|2\.5-pro", mid))
    coding = 1.0 if re.search(r"coder|codestral|devstral|(^|[/-])code", mid) else (0.5 if "coding" in caps else 0.0)
    vision = "vision" in caps or bool(re.search(r"vision|4o|gpt-4\.1|gpt-5|gemini|claude|pixtral|llama-4|-vl", mid))
    penalty = 0.0
    if re.search(r"preview|[-/]exp|beta", mid):
        penalty -= 1.0
    if re.search(r"-\d{4}-\d{2}-\d{2}$", mid):
        penalty -= 0.3
    return dict(tier=tier, reasoning=reasoning, coding=coding, vision=vision, penalty=penalty,
                version=_version(mid), context=model.context_window)


def _context_score(window):
    if not window:
        return 0.4  # unknown: assume middling rather than best or worst
    return max(0.0, min(1.0, math.log10(max(window, 1) / 8000) / math.log10(1_000_000 / 8000)))


def task_score(task, model):
    w = TASK_PROFILE.get(task, TASK_PROFILE["General chat"])
    pr = model_profile(model)
    return (10 - 2.5 * abs(pr["tier"] - w["tier"])
            + w["reasoning"] * (1.0 if pr["reasoning"] else 0.0)
            + w["coding"] * pr["coding"]
            + w["context"] * _context_score(pr["context"])
            + w["vision"] * (1.0 if pr["vision"] else 0.0)
            + pr["penalty"])


def explain_pick(task, model):
    pr = model_profile(model)
    bits = [f"{TIER_LABELS.get(int(round(pr['tier'])), 'standard')} tier"]
    if pr["reasoning"]:
        bits.append("reasoning-capable")
    if pr["coding"] >= 1:
        bits.append("code-specialised")
    if pr["vision"]:
        bits.append("vision")
    if model.context_window:
        bits.append(f"{model.context_window // 1000}K context")
    billing = getattr(model, "billing", "unknown")
    if billing in ("free", "paid"):
        bits.append("free model" if billing == "free" else "billed model")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# Live catalog support: let auto mode see models the provider reports today,
# not just the hardcoded starter list.
# ---------------------------------------------------------------------------
_NON_CHAT = re.compile(
    r"embed|moderation|whisper|tts|transcribe|speech|audio|realtime|image|imagen|dall-e|sora|veo|video|"
    r"rerank|guard|safety|search|computer-use|codex|deep-research|davinci|babbage|curie|(^|[/-])ada\b|"
    r"[-/]live|learnlm|aqa|robotics")
_DATED = re.compile(r"^(.*)-\d{4}-\d{2}-\d{2}$")


def is_chat_candidate(provider_id, model_id):
    mid = model_id.lower()
    if _NON_CHAT.search(mid):
        return False
    if provider_id == "openai" and re.search(r"-pro\b", mid):
        return False  # OpenAI pro models are Responses-API only, not chat completions
    return True


class ProviderView:
    """A provider whose model list comes from the live catalog but whose chat() is the real adapter's."""

    def __init__(self, provider, models):
        self._provider = provider
        self._models = models

    def list_models(self):
        return self._models

    def __getattr__(self, name):
        return getattr(self._provider, name)


def live_provider_views(providers, catalog):
    """Swap in live model lists where the catalog verified them; keep starter lists otherwise."""
    from .models import ModelInfo
    live = {e.get("provider_id"): e for e in (catalog or []) if e.get("catalog_source") == "live"}
    views = []
    for provider in providers:
        entry = live.get(provider.provider_id)
        if not entry:
            views.append(provider)
            continue
        starter = {m.model_id: m for m in provider.list_models()}
        ids = {m.get("model_id") for m in entry.get("models", [])}
        models = []
        for m in entry.get("models", []):
            mid = m.get("model_id")
            if not mid or not is_chat_candidate(provider.provider_id, mid):
                continue
            dated = _DATED.match(mid)
            if dated and dated.group(1) in ids:
                continue  # an undated alias exists; skip the pinned snapshot
            known = starter.get(mid)
            caps = list(dict.fromkeys([*(known.capabilities if known else []), *m.get("capabilities", [])]))
            if "text" not in caps:
                caps.append("text")
            models.append(ModelInfo(provider.display_name, mid, m.get("display_name") or mid,
                                    m.get("context_window") or (known.context_window if known else None),
                                    caps, "live catalog", modality_support=m.get("modality_support")))
        views.append(ProviderView(provider, models) if models else provider)
    return views


@dataclass
class Selection:
    provider: object | None
    model_id: str | None
    reason: str


FREE_BONUS = 4.0  # with "prefer free", a free model beats a paid one that is up to ~1.5 tiers better


def rank_models(providers, task, prefer_provider=None, prefer_free=False, load=None):
    """Rank every model for a task, best first.

    Scores within a quarter point count as equal. Equal models are ordered by: the provider you have used
    least today (``load`` = tokens used today per provider, which spreads usage across your plans), then the
    provider you selected, then context window, then model version.
    """
    required = TASKS.get(task, {"text"})
    load = load or {}
    candidates = []
    for provider in providers:
        for model in provider.list_models():
            caps = {str(c).lower() for c in model.capabilities}
            # modality-specific generation must be declared explicitly, never inferred
            if task in ("Image generation", "Video generation"):
                if not (required & caps):
                    continue
                score = 0.0
            else:
                if "text" not in caps and not (required & caps):
                    continue
                score = task_score(task, model)
                if prefer_free and getattr(model, "billing", "unknown") == "free":
                    score += FREE_BONUS
            candidates.append((score, provider, model))
    candidates.sort(key=lambda c: (round(c[0] * 4),
                                   -load.get(c[1].provider_id, 0),
                                   c[1].provider_id == prefer_provider,
                                   round(_context_score(c[2].context_window), 1),
                                   model_profile(c[2])["version"]),
                    reverse=True)
    return candidates


MEDIA_TASKS = ("Image generation", "Video generation")


def _media_unavailable(task):
    if task == "Video generation":
        return Selection(None, None,
            "Video generation is not available yet: this app has no adapter for any provider's video API, "
            "and a chat model cannot produce video by itself.")
    return Selection(None, None,
        "No model is known to generate images yet. Select a Google Gemini model and send your request: "
        "the app will try it and remember the result. Image output is only supported for Gemini so far.")


def choose_model(providers, task, configured_provider=None, configured_model=None, auto=False, has_key=None,
                 prefer_free=False, load=None):
    """Pick the provider/model for a request.

    * Manual (default): honour the provider/model the user selected. Any model id is accepted,
      so models found by live catalog discovery work even if they are not in the starter list.
    * auto=True: ignore the selection and use the best-ranked model for the task, considering
      only providers that have a saved key (``has_key(provider_id) -> bool``).
    * Media tasks need a model that explicitly declares the capability, otherwise unavailable.
    """
    ranked = rank_models(providers, task, None if auto else configured_provider, prefer_free, load)

    if task in MEDIA_TASKS:
        if not ranked:
            return _media_unavailable(task)
        for _, p, m in ranked:
            if p.provider_id == configured_provider and m.model_id == configured_model:
                return Selection(p, m.model_id, f"Using the selected {p.display_name} / {m.model_id}.")
        _, p, m = ranked[0]
        return Selection(p, m.model_id, f"Selected {p.display_name} / {m.model_id} for {task.lower()}.")

    if auto or not configured_provider:
        pool = ranked
        if has_key is not None:
            pool = [r for r in ranked if has_key(r[1].provider_id)]
        if not pool:
            if has_key is not None and ranked:
                return Selection(None, None,
                    f"No provider with a saved API key offers a model for {task}. Add a saved API key in Settings.")
            return Selection(None, None, f"No configured model supports {task}.")
        _, p, m = pool[0]
        return Selection(p, m.model_id,
            f"Auto-selected {p.display_name} / {m.model_id} for {task}: {explain_pick(task, m)}.")

    provider = next((p for p in providers if p.provider_id == configured_provider), None)
    if provider is None:
        return Selection(None, None, f"Unknown provider '{configured_provider}'.")
    model_id = configured_model
    if not model_id:
        best = next((m for _, p, m in ranked if p is provider), None)
        if best is None:
            models = provider.list_models()
            best = models[0] if models else None
        if best is None:
            return Selection(None, None, f"{provider.display_name} has no models configured.")
        model_id = best.model_id
    return Selection(provider, model_id, f"Using the selected {provider.display_name} / {model_id}.")


def estimate_cost_usd(input_tokens, output_tokens, input_per_million, output_per_million):
    if input_per_million is None or output_per_million is None:
        return None
    return ((max(0, input_tokens or 0) * float(input_per_million)) +
            (max(0, output_tokens or 0) * float(output_per_million))) / 1_000_000


def estimate_input_tokens(messages):
    text = "".join(str(m.get("content", "")) for m in messages)
    return max(1, (len(text) + 3) // 4) if text else 0
