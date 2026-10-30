"""User-added models, free/paid classification, and routing helpers.

Everything here is local configuration stored next to the telemetry data. No secrets are stored.

Free vs paid is classified from the best evidence available, in this order:
  1. a choice you made in the catalog ("set by you")
  2. the billing you picked when you added the model
  3. provider-published pricing (OpenRouter returns per-token prices)
  4. a ":free" suffix in the model id (OpenRouter convention)
  5. per-model pricing you configured under budgets (both rates 0 -> free)
  otherwise the model stays "unknown". Nothing is assumed from the vendor name, because plans, free tiers
  and promotions change; "unknown" models are never blocked, they only rank after known-free ones when
  "Prefer free" is on.
"""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping

from .capabilities import CHECK_PREFIX, MEDIA_KINDS, modality_status
from .models import ModelInfo
from .telemetry import connect, get_spend_config

BILLING_VALUES = ("free", "paid", "unknown")
CAPABILITY_CHOICES = ("text", "coding", "analysis", "vision", "fast", "image_generation")
# Providers whose adapter in this app can carry an image back to the chat window. This is about the APP,
# not the model: a model may be able to draw while its provider's adapter here cannot return pictures yet.
IMAGE_OUTPUT_ADAPTERS = {"gemini"}
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,199}$")


def _con():
    con = connect()
    con.execute("""CREATE TABLE IF NOT EXISTS custom_models (
      provider_id TEXT NOT NULL, model_id TEXT NOT NULL, display_name TEXT NOT NULL,
      capabilities TEXT NOT NULL, context_window INTEGER, billing TEXT NOT NULL DEFAULT 'unknown',
      created_at TEXT NOT NULL, PRIMARY KEY (provider_id, model_id))""")
    con.execute("""CREATE TABLE IF NOT EXISTS capability_checks (
      provider_id TEXT NOT NULL, model_id TEXT NOT NULL, capability TEXT NOT NULL,
      result INTEGER NOT NULL, evidence TEXT NOT NULL, checked_at TEXT NOT NULL,
      PRIMARY KEY (provider_id, model_id, capability))""")
    con.execute("""CREATE TABLE IF NOT EXISTS model_billing (
      provider_id TEXT NOT NULL, model_id TEXT NOT NULL, billing TEXT NOT NULL,
      PRIMARY KEY (provider_id, model_id))""")
    return con


# ----------------------------------------------------------------------------- storage
def list_custom_models(provider_id: str | None = None) -> list[dict[str, Any]]:
    import json
    con = _con()
    try:
        sql = "SELECT * FROM custom_models" + (" WHERE provider_id=?" if provider_id else "") + " ORDER BY created_at"
        rows = con.execute(sql, (provider_id,) if provider_id else ()).fetchall()
        return [dict(model_id=r["model_id"], provider_id=r["provider_id"], display_name=r["display_name"],
                     capabilities=json.loads(r["capabilities"]), context_window=r["context_window"],
                     billing=r["billing"]) for r in rows]
    finally:
        con.close()


def add_custom_model(provider_id: str, model_id: str, display_name: str = "",
                     capabilities: list[str] | None = None, context_window: int | None = None,
                     billing: str = "unknown") -> None:
    import json
    model_id = (model_id or "").strip()
    if not MODEL_ID_RE.match(model_id):
        raise ValueError("Model ID may only contain letters, digits and . _ : / @ + - (max 200 characters).")
    caps = [c for c in dict.fromkeys(capabilities or ["text"])]
    bad = [c for c in caps if c not in CAPABILITY_CHOICES]
    if bad:
        raise ValueError(f"Unsupported capability: {', '.join(bad)}. Allowed: {', '.join(CAPABILITY_CHOICES)}.")
    if "text" not in caps:
        caps.insert(0, "text")
    if billing not in BILLING_VALUES:
        raise ValueError("Billing must be free, paid or unknown.")
    if context_window is not None and not (1_000 <= context_window <= 100_000_000):
        raise ValueError("Context window must be between 1,000 and 100,000,000 tokens.")
    name = (display_name or "").strip()[:120] or model_id
    con = _con()
    try:
        con.execute("""INSERT INTO custom_models(provider_id, model_id, display_name, capabilities, context_window, billing, created_at)
                       VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(provider_id, model_id) DO UPDATE SET display_name=excluded.display_name,
                       capabilities=excluded.capabilities, context_window=excluded.context_window, billing=excluded.billing""",
                    (provider_id, model_id, name, json.dumps(caps), context_window, billing,
                     datetime.now(timezone.utc).isoformat()))
        con.commit()
    finally:
        con.close()


def delete_custom_model(provider_id: str, model_id: str) -> bool:
    con = _con()
    try:
        cur = con.execute("DELETE FROM custom_models WHERE provider_id=? AND model_id=?", (provider_id, model_id))
        con.execute("DELETE FROM model_billing WHERE provider_id=? AND model_id=?", (provider_id, model_id))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def set_billing_override(provider_id: str, model_id: str, billing: str) -> None:
    """billing: 'free' | 'paid' | 'auto' (auto removes your override)."""
    if billing not in ("free", "paid", "auto"):
        raise ValueError("Billing must be free, paid or auto.")
    con = _con()
    try:
        if billing == "auto":
            con.execute("DELETE FROM model_billing WHERE provider_id=? AND model_id=?", (provider_id, model_id))
        else:
            con.execute("""INSERT INTO model_billing(provider_id, model_id, billing) VALUES(?,?,?)
                           ON CONFLICT(provider_id, model_id) DO UPDATE SET billing=excluded.billing""",
                        (provider_id, model_id, billing))
        con.commit()
    finally:
        con.close()


def record_capability_check(provider_id: str, model_id: str, capability: str, result: bool, evidence: str) -> None:
    """Remember what actually happened when a model was asked to do something (for example return an image)."""
    con = _con()
    try:
        con.execute("""INSERT INTO capability_checks(provider_id, model_id, capability, result, evidence, checked_at)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(provider_id, model_id, capability) DO UPDATE SET result=excluded.result,
                       evidence=excluded.evidence, checked_at=excluded.checked_at""",
                    (provider_id, model_id, capability, 1 if result else 0, evidence[:300],
                     datetime.now(timezone.utc).isoformat()))
        con.commit()
    finally:
        con.close()


def get_capability_checks() -> dict[tuple[str, str, str], dict[str, Any]]:
    con = _con()
    try:
        return {(r["provider_id"], r["model_id"], r["capability"]):
                {"result": bool(r["result"]), "evidence": r["evidence"], "checked_at": r["checked_at"]}
                for r in con.execute("SELECT * FROM capability_checks")}
    finally:
        con.close()


def image_output_status(provider_id: str, model_id: str, capabilities, checks=None) -> dict[str, Any]:
    """verified | failed | declared | untested | unsupported_adapter, with the evidence behind it."""
    if provider_id not in IMAGE_OUTPUT_ADAPTERS:
        return {"status": "unsupported_adapter",
                "detail": "This app has no image-output adapter for this provider yet."}
    check = (checks if checks is not None else get_capability_checks()).get((provider_id, model_id, "image_generation"))
    if check:
        day = check["checked_at"][:10]
        return {"status": "verified" if check["result"] else "failed", "detail": f"{check['evidence']} ({day})"}
    if "image_generation" in capabilities:
        return {"status": "declared", "detail": "You marked this model as image-capable; not tested yet."}
    return {"status": "untested", "detail": "Not tested. Pick the Image generation task or press Test to find out."}


def _apply_checks(capabilities: list[str], provider_id: str, model_id: str, checks) -> list[str]:
    """A real test result beats a declaration: success adds image_generation, failure removes it."""
    check = checks.get((provider_id, model_id, "image_generation"))
    caps = [c for c in capabilities]
    if check and check["result"] and "image_generation" not in caps:
        caps.append("image_generation")
    if check and not check["result"]:
        caps = [c for c in caps if c != "image_generation"]
    return caps


def _input_support(pid: str, mid: str, declared, caps, checks) -> dict[str, dict[str, str]]:
    """Per input kind: yes / no / unlikely / unknown plus where that knowledge came from (shown in the catalog)."""
    out = {}
    for kind in MEDIA_KINDS:
        status, source = modality_status(kind, provider_id=pid, model_id=mid, declared=declared, user_caps=caps, checks=checks)
        check = (checks or {}).get((pid, mid, CHECK_PREFIX + kind))
        out[kind] = {"status": status, "source": source,
                     "evidence": check["evidence"] if check else "", "checked_at": check["checked_at"] if check else ""}
    return out


def _probe_info(pid: str, mid: str, checks) -> dict[str, dict[str, str]]:
    """What Model Lab learned about plain text / code answers and image output for this model."""
    out = {}
    for name, cap in (("text", "works:text"), ("code", "works:code"), ("image_output", "image_generation")):
        check = (checks or {}).get((pid, mid, cap))
        out[name] = ({"status": "yes" if check["result"] else "no", "evidence": check["evidence"],
                      "checked_at": check["checked_at"]} if check else {"status": "unknown", "evidence": "", "checked_at": ""})
    return out


def _overrides() -> dict[tuple[str, str], str]:
    con = _con()
    try:
        return {(r["provider_id"], r["model_id"]): r["billing"] for r in con.execute("SELECT * FROM model_billing")}
    finally:
        con.close()


def provider_load_today() -> dict[str, int]:
    """Tokens used today per provider, from local telemetry. Used to spread load when models tie."""
    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    con = _con()
    try:
        rows = con.execute("SELECT provider_id, COALESCE(SUM(total_tokens),0) AS t FROM usage_events "
                           "WHERE ts>=? GROUP BY provider_id", (day_start,)).fetchall()
        return {r["provider_id"]: int(r["t"] or 0) for r in rows}
    finally:
        con.close()


# ----------------------------------------------------------------------------- classification
def _rate(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_billing(provider_id: str, model_id: str, pricing: Mapping | None = None,
                     overrides: Mapping | None = None, custom_billing: str | None = None,
                     config: Mapping | None = None) -> tuple[str, str]:
    """Return (billing, source). Never guesses beyond the evidence listed in the module docstring."""
    overrides = overrides or {}
    chosen = overrides.get((provider_id, model_id))
    if chosen in ("free", "paid"):
        return chosen, "set by you"
    if custom_billing in ("free", "paid"):
        return custom_billing, "set when you added it"
    if isinstance(pricing, Mapping):
        prompt, completion = _rate(pricing.get("prompt")), _rate(pricing.get("completion"))
        if prompt is not None and completion is not None and prompt >= 0 and completion >= 0:
            return ("free", "provider pricing data") if prompt == 0 and completion == 0 else ("paid", "provider pricing data")
    if model_id.lower().endswith(":free"):
        return "free", "model id suffix :free"
    cfg = ((config or {}).get("pricing") or {}).get(provider_id, {}).get(model_id)
    if isinstance(cfg, Mapping):
        a, b = _rate(cfg.get("input_per_million_usd")), _rate(cfg.get("output_per_million_usd"))
        if a is not None and b is not None and a >= 0 and b >= 0:
            return ("free", "your configured pricing") if a == 0 and b == 0 else ("paid", "your configured pricing")
    return "unknown", "not classified"


def enrich_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add billing info to every catalog model and merge in the models you added."""
    overrides, config, checks = _overrides(), get_spend_config(), get_capability_checks()
    customs: dict[str, list[dict]] = {}
    for c in list_custom_models():
        customs.setdefault(c["provider_id"], []).append(c)
    result = []
    for entry in catalog:
        pid = entry.get("provider_id")
        entry = dict(entry)
        added = {c["model_id"]: c for c in customs.get(pid, [])}
        models = []
        for m in entry.get("models", []):
            m = dict(m)
            custom = added.pop(m.get("model_id"), None)
            if custom:
                m["capabilities"] = list(dict.fromkeys([*m.get("capabilities", []), *custom["capabilities"]]))
                m["custom"] = True
            billing, source = classify_billing(pid, m.get("model_id", ""), m.get("pricing"), overrides,
                                               custom["billing"] if custom else None, config)
            m["capabilities"] = _apply_checks(m.get("capabilities", []), pid, m.get("model_id", ""), checks)
            m.update(billing=billing, billing_source=source,
                     input_support=_input_support(pid, m.get("model_id", ""), m.get("modality_support"), m["capabilities"], checks),
                     probes=_probe_info(pid, m.get("model_id", ""), checks),
                     image_output=image_output_status(pid, m.get("model_id", ""), m["capabilities"], checks))
            m.setdefault("custom", False)
            models.append(m)
        for c in added.values():  # models you added that the provider list doesn't contain
            billing, source = classify_billing(pid, c["model_id"], None, overrides, c["billing"], config)
            caps = _apply_checks(c["capabilities"], pid, c["model_id"], checks)
            models.append(dict(model_id=c["model_id"], display_name=c["display_name"], capabilities=caps,
                               context_window=c["context_window"], metadata_source="added by you",
                               billing=billing, billing_source=source, custom=True, pricing=None,
                               input_support=_input_support(pid, c["model_id"], None, caps, checks),
                               probes=_probe_info(pid, c["model_id"], checks),
                               image_output=image_output_status(pid, c["model_id"], caps, checks)))
        entry["models"] = models
        result.append(entry)
    return result


# ----------------------------------------------------------------------------- routing
def routing_providers(providers: list, catalog: list[dict[str, Any]] | None = None) -> list:
    """Wrap providers so routing sees billing info and the models you added.

    `providers` may already be live views. The returned objects delegate chat() to the real adapters.
    """
    from .task_routing import ProviderView

    overrides, config, checks = _overrides(), get_spend_config(), get_capability_checks()
    pricing = {(e.get("provider_id"), m.get("model_id")): m.get("pricing")
               for e in (catalog or []) for m in e.get("models", [])}
    wrapped = []
    for provider in providers:
        pid = provider.provider_id
        customs = {c["model_id"]: c for c in list_custom_models(pid)}
        models: dict[str, ModelInfo] = {}
        for m in provider.list_models():
            custom = customs.get(m.model_id)
            caps = list(dict.fromkeys([*m.capabilities, *(custom["capabilities"] if custom else [])]))
            caps = _apply_checks(caps, pid, m.model_id, checks)
            billing, _ = classify_billing(pid, m.model_id, pricing.get((pid, m.model_id)), overrides,
                                          custom["billing"] if custom else None, config)
            models[m.model_id] = replace(m, capabilities=caps, billing=billing)
        for mid, c in customs.items():
            if mid not in models:
                billing, _ = classify_billing(pid, mid, None, overrides, c["billing"], config)
                models[mid] = ModelInfo(provider.display_name, mid, c["display_name"], c["context_window"],
                                        _apply_checks(list(c["capabilities"]), pid, mid, checks), "added by you", billing)
        wrapped.append(ProviderView(provider, list(models.values())))
    return wrapped
