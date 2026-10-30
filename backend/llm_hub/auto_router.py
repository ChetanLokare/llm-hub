"""Auto-pick and Safe fallback, driven by evidence instead of model names.

How a request is handled
------------------------
1. The request is examined for what it CARRIES: images, PDFs, audio, video (zips are opened first, see
   attachments.py). Text and documents that can be turned into text need nothing special.
2. Every model the connected providers report today becomes a candidate. A candidate is dropped only when
   there is positive evidence it cannot take an input the request needs (the provider's own model list says
   so, or a recent real request was rejected). "We don't know" is NOT a reason to drop a model.
3. Candidates are ordered with no hand-written ranking of vendors or model names:
     a. a model you picked yourself, when Auto-pick is off      (your choice always goes first)
     b. models that did not just fail (rate-limited / dead key / missing model go last for a while, and so do
        models that failed a Model Lab text test)
     c. with "Prefer free": free first, then unknown, then paid  (a soft order - nothing is excluded)
     d. confirmed support for the needed inputs first, then untested, then "recently rejected" last
     e. models that passed a Model Lab text test, then untested ones
     f. models whose published context window fits the request before those that are too small
     g. then providers are interleaved - one model per provider per round - and within a round the provider
        used least today / least recently goes first, so no vendor is favoured
4. Safe fallback walks that list until one model answers. ANY failure moves on to the next candidate
   (rate limit, server error, timeout, bad key, retired model, input rejected, too large...). Each result is
   remembered: a success marks the model as able to take that input, a clear rejection marks it as unable.
5. If every model rejects the attachments, the request is retried once as text only, with a plain note
   telling the model (and you) which files could not be read, so you still get an answer.

Trade-off, stated plainly: retrying after a timeout can occasionally bill twice for one request. With Safe
fallback on, finishing the task is treated as more important than that risk.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .attachments import Prepared, media_kinds_in, render_user_content
from .capabilities import (CHECK_PREFIX, HEALTH, RECENT_USE, Health, failure_kind, modality_status, short_error,
                           works_status)
from .models import ModelInfo

BILLING_RANK = {"free": 0, "unknown": 1, "paid": 2}


class BudgetBlocked(Exception):
    def __init__(self, detail: dict | str):
        super().__init__(str(detail))
        self.detail = detail


class MissingKey(Exception):
    pass


class NoCandidates(Exception):
    pass


class AllAttemptsFailed(Exception):
    def __init__(self, log: list[dict], last: BaseException | None):
        super().__init__("; ".join(f"{e['label']}: {e['error']}" for e in log) or "no model could be tried")
        self.log, self.last = log, last


@dataclass
class Attempt:
    provider: Any
    model: ModelInfo
    statuses: dict = field(default_factory=dict)   # kind -> (status, source)
    billing: str = "unknown"
    preferred: bool = False
    tier: tuple = ()

    @property
    def label(self) -> str:
        return f"{self.provider.display_name} / {self.model.model_id}"


@dataclass
class ChatOutcome:
    answer: str
    usage: Any
    images: list
    provider: Any
    model_id: str
    context_note: str
    fallback_reason: str = ""
    routing_reason: str = ""
    attempts: list = field(default_factory=list)
    degraded: bool = False
    skipped: list = field(default_factory=list)


# ----------------------------------------------------------------------------- planning
def plan_attempts(views, required: set[str], optional: set[str], est_tokens: int, checks, *, has_key,
                  prefer_free: bool = False, preferred: tuple[str, str] | None = None, load: dict | None = None,
                  health: Health | None = None, recent: dict | None = None, rng: random.Random | None = None,
                  now=None) -> list[Attempt]:
    health = health if health is not None else HEALTH
    recent = recent if recent is not None else RECENT_USE
    rng = rng or random.Random()
    load = load or {}
    kinds = sorted(required | optional)
    pool: list[Attempt] = []
    seen: set[tuple[str, str]] = set()

    def make(provider, model, is_preferred):
        statuses = {k: modality_status(k, provider_id=provider.provider_id, model_id=model.model_id,
                                       declared=getattr(model, "modality_support", None),
                                       user_caps=model.capabilities, checks=checks, now=now) for k in kinds}
        billing = getattr(model, "billing", "unknown") or "unknown"
        window = model.context_window
        fits = 0 if (not window or window * 0.9 >= est_tokens) else 1
        needed = [statuses[k][0] for k in required]
        confirmed = 2 if "unlikely" in needed else (0 if all(x == "yes" for x in needed) else 1)
        works = works_status(provider.provider_id, model.model_id, checks, now)
        tier = (0 if is_preferred else 1,
                1 if (health.is_cooling(provider.provider_id, model.model_id) or works == "unlikely") else 0,
                BILLING_RANK.get(billing, 1) if prefer_free else 0,
                confirmed, 0 if works == "yes" else 1, fits)
        return Attempt(provider, model, statuses, billing, is_preferred, tier)

    for provider in views:
        for model in provider.list_models():
            key = (provider.provider_id, model.model_id)
            if key in seen:
                continue
            seen.add(key)
            is_preferred = preferred == key
            if not is_preferred:
                if not has_key(provider.provider_id):
                    continue
            attempt = make(provider, model, is_preferred)
            if not is_preferred and any(attempt.statuses[k][0] == "no" for k in required):
                continue
            pool.append(attempt)

    # A model you typed or picked that the provider's list does not contain is still honoured.
    if preferred and (preferred not in seen):
        provider = next((v for v in views if v.provider_id == preferred[0]), None)
        if provider is not None:
            pool.append(make(provider, ModelInfo(provider.display_name, preferred[1], preferred[1], None, ["text"],
                                                 "selected by you"), True))

    # order inside each tier: larger published context first, randomly among equals; then interleave providers
    for a in pool:
        a._rand = rng.random()  # type: ignore[attr-defined]
    pool.sort(key=lambda a: (a.tier, -(a.model.context_window or 0), a._rand))  # type: ignore[attr-defined]
    index: dict[tuple, int] = defaultdict(int)
    keyed = []
    for a in pool:
        slot = (a.tier, a.provider.provider_id)
        keyed.append((a.tier, index[slot], load.get(a.provider.provider_id, 0),
                      recent.get(a.provider.provider_id, 0.0), a._rand, a))  # type: ignore[attr-defined]
        index[slot] += 1
    keyed.sort(key=lambda t: t[:5])
    return [t[5] for t in keyed]


def explain(attempt: Attempt, required: set[str], auto: bool) -> str:
    bits = []
    for k in sorted(required):
        status, source = attempt.statuses.get(k, ("unknown", ""))
        bits.append(f"{k} input confirmed ({source})" if status == "yes"
                    else f"{k} input untested - trying it" if status == "unknown"
                    else f"{k} input was rejected recently - last resort")
    if attempt.billing in ("free", "paid"):
        bits.append("free model" if attempt.billing == "free" else "billed model")
    if attempt.model.context_window:
        bits.append(f"{attempt.model.context_window // 1000}K context")
    prefix = "Auto-selected" if auto else "Using the selected"
    return f"{prefix} {attempt.label}" + (": " + ", ".join(bits) if bits else "") + "."


def no_candidates_message(views, required: set[str], has_key) -> str:
    keyed = [v for v in views if has_key(v.provider_id)]
    if not keyed:
        return "No provider with a saved API key is available. Add a key in Settings."
    if required:
        return ("No connected model accepts " + " + ".join(sorted(required)) + " input: every model's provider "
                "listing or a recent test says it can't. Add a key for another provider, or remove that attachment.")
    return "The connected providers reported no usable chat models."


# ----------------------------------------------------------------------------- running
def _build_messages(history: list[dict], text: str, prepared: Prepared, native_ok,
                    earlier: dict[int, Prepared] | None = None) -> tuple[list[dict], set[str]]:
    content = render_user_content(text, prepared, native_ok)
    messages = [dict(m) for m in history[:-1]] if history and history[-1].get("role") == "user" else [dict(m) for m in history]
    used = set(media_kinds_in(content))
    for idx, prep in (earlier or {}).items():
        if 0 <= idx < len(messages):
            rendered = render_user_content(messages[idx]["content"], prep, native_ok)
            messages[idx]["content"] = rendered
            used |= media_kinds_in(rendered)
    messages.append({"role": "user", "content": content})
    return messages, used


def run_chat(*, views, history: list[dict], prepared: Prepared, selected: tuple[str, str] | None, auto: bool,
             safe_fallback: bool, prefer_free: bool, has_key: Callable[[str], bool],
             send: Callable[[Any, ModelInfo, list[dict]], tuple], checks: dict, record_check: Callable,
             load: dict | None = None, budget_gate: Callable | None = None, health: Health | None = None,
             rng: random.Random | None = None, max_attempts: int = 12, deadline_s: float = 180.0,
             clock: Callable[[], float] = time.monotonic, earlier: dict[int, Prepared] | None = None) -> ChatOutcome:
    """Run one chat request. ``send(provider, model, messages) -> (answer, usage, images, context_note)``."""
    health = health if health is not None else HEALTH
    text = history[-1]["content"] if history and history[-1].get("role") == "user" else ""
    preferred = None if (auto or not selected) else selected

    earlier = earlier or {}

    def required_of(prep: Prepared, earl: dict[int, Prepared]) -> set[str]:
        out = set(prep.native_required())
        for e in earl.values():
            out |= e.native_required()
        return out

    def optional_of(prep: Prepared, earl: dict[int, Prepared]) -> set[str]:
        out = set(prep.native_optional())
        for e in earl.values():
            out |= e.native_optional()
        return out

    def attempt_plan(prep: Prepared, earl: dict[int, Prepared]) -> list[Attempt]:
        est = (prep.estimated_tokens() + sum(e.estimated_tokens() for e in earl.values())
               + sum(len(str(m.get("content", ""))) for m in history) // 4)
        return plan_attempts(views, required_of(prep, earl), optional_of(prep, earl), est, checks, has_key=has_key,
                             prefer_free=prefer_free, preferred=preferred, load=load, health=health, rng=rng)

    def execute(attempts: list[Attempt], prep: Prepared, earl: dict[int, Prepared]) -> ChatOutcome:
        required = required_of(prep, earl)
        log: list[dict] = []
        started, last_exc = clock(), None
        dead_providers: set[str] = set()
        for n, att in enumerate(attempts):
            if n > 0:
                if not safe_fallback or n >= max_attempts or clock() - started > deadline_s:
                    break
            if att.provider.provider_id in dead_providers:
                continue

            def native_ok(kind, att=att):
                status = att.statuses.get(kind, ("unknown", ""))[0]
                return status == "yes" or (status == "unknown" and kind in required)

            messages, used = _build_messages(history, text, prep, native_ok, earl)
            try:
                if budget_gate:
                    budget_gate(att.provider.provider_id, att.model.model_id, messages)
                answer, usage, images, note = send(att.provider, att.model, messages)
                if safe_fallback and not (answer or images):
                    # An empty reply (often a safety block) is not an answer: try the next model.
                    raise RuntimeError("The model returned no content (possibly blocked by a safety filter).")
            except (BudgetBlocked, MissingKey) as exc:
                log.append({"label": att.label, "error": str(exc.detail if isinstance(exc, BudgetBlocked) else exc), "kind": "skipped"})
                last_exc = exc
                if not safe_fallback:
                    raise
                continue
            except Exception as exc:  # noqa: BLE001 - every provider/SDK error type is handled the same way
                kind = failure_kind(exc)
                log.append({"label": att.label, "error": short_error(exc), "kind": kind})
                last_exc = exc
                health.penalize(att.provider.provider_id, att.model.model_id, kind)
                if used and kind in ("modality", "bad_request") and (len(used) == 1 or kind == "modality"):
                    blamed = [getattr(exc, "kind", None)] if getattr(exc, "kind", None) in used else sorted(used)
                    for k in blamed:
                        record_check(att.provider.provider_id, att.model.model_id, CHECK_PREFIX + k, False,
                                     f"Rejected {k} input: {short_error(exc, 120)}")
                if kind == "auth":
                    dead_providers.add(att.provider.provider_id)
                if not safe_fallback:
                    raise
                continue

            for k in used:
                record_check(att.provider.provider_id, att.model.model_id, CHECK_PREFIX + k, True,
                             f"Answered a request containing {k} input")
            RECENT_USE[att.provider.provider_id] = clock()
            reason = ""
            if log:
                first = log[0]
                reason = (f"{first['label']} failed ({first['kind'].replace('_', ' ')}: {first['error']}); "
                          f"switched to {att.label}" + (f" after {len(log)} failed attempt(s)." if len(log) > 1 else "."))
            return ChatOutcome(answer, usage, images, att.provider, att.model.model_id, note, reason,
                               explain(att, required, auto), log)
        raise AllAttemptsFailed(log, last_exc)

    needed = required_of(prepared, earlier)
    attempts = attempt_plan(prepared, earlier)
    if attempts:
        try:
            return execute(attempts, prepared, earlier)
        except AllAttemptsFailed:
            if not (safe_fallback and needed):
                raise
    elif not (safe_fallback and needed):
        raise NoCandidates(no_candidates_message(views, needed, has_key))

    # Nothing accepted the files natively: answer from the text the files contain, and say what was left out.
    degraded = prepared.text_only()
    degraded_earlier = {i: e.text_only() for i, e in earlier.items()}
    attempts = attempt_plan(degraded, degraded_earlier)
    if not attempts:
        raise NoCandidates(no_candidates_message(views, set(), has_key))
    outcome = execute(attempts, degraded, degraded_earlier)
    outcome.degraded = True
    outcome.skipped = [p.name for p in [*prepared.parts, *(q for e in earlier.values() for q in e.parts)]
                       if p.kind != "text" and p.text is None]
    note = ("These files could not be read natively by any available model and were left out: "
            + ", ".join(outcome.skipped) + ".") if outcome.skipped else ""
    outcome.fallback_reason = (outcome.fallback_reason + " " + note).strip()
    return outcome
