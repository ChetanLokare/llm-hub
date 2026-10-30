"""Local usage telemetry and estimate calibration for LLM Hub.

Quota values are only reported when a provider exposes them through a verified API.
This MVP logs token usage returned by adapters and never guesses remaining quota.
"""
import os, sqlite3, json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("LLMHUB_TELEMETRY_DB", Path.home() / ".llm_hub" / "telemetry.sqlite3"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS usage_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, provider_id TEXT NOT NULL,
      model_id TEXT NOT NULL, task TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER,
      total_tokens INTEGER, usage_source TEXT NOT NULL, estimated_input_tokens INTEGER,
      estimated_cost_usd REAL, request_ok INTEGER NOT NULL DEFAULT 1)""")
    con.execute("""CREATE TABLE IF NOT EXISTS spend_config (
      config_key TEXT PRIMARY KEY, config_value TEXT NOT NULL)""")
    return con

DEFAULT_BUDGETS = {"daily_limit_usd": None, "monthly_limit_usd": None, "provider_limits_usd": {}, "pricing": {}}

def get_spend_config():
    con = connect()
    try:
        rows = {r["config_key"]: json.loads(r["config_value"]) for r in con.execute("SELECT config_key, config_value FROM spend_config")}
        return rows.get("budgets", DEFAULT_BUDGETS.copy())
    finally:
        con.close()

def save_spend_config(config):
    # Validate shape and numeric values at the API boundary; persist only configuration, never secrets.
    con = connect()
    try:
        con.execute("INSERT INTO spend_config(config_key,config_value) VALUES('budgets',?) ON CONFLICT(config_key) DO UPDATE SET config_value=excluded.config_value", (json.dumps(config),))
        con.commit()
    finally:
        con.close()

def spend_totals(provider_id=None):
    con = connect()
    try:
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        where = " WHERE provider_id=?" if provider_id else ""
        args = (provider_id,) if provider_id else ()
        day = con.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM usage_events WHERE ts>=?" + (" AND provider_id=?" if provider_id else ""), (day_start, *args)).fetchone()[0]
        month = con.execute("SELECT COALESCE(SUM(estimated_cost_usd),0) FROM usage_events WHERE ts>=?" + (" AND provider_id=?" if provider_id else ""), (month_start, *args)).fetchone()[0]
        return {"today_estimated_cost_usd": float(day or 0), "month_estimated_cost_usd": float(month or 0)}
    finally:
        con.close()

def estimate_configured_cost(provider_id, model_id, input_tokens, output_tokens, config):
    pricing = config.get("pricing", {}).get(provider_id, {}).get(model_id)
    if not isinstance(pricing, dict):
        return None
    try:
        in_rate = float(pricing["input_per_million_usd"])
        out_rate = float(pricing["output_per_million_usd"])
        if min(in_rate, out_rate) < 0:
            return None
        return (max(0, input_tokens or 0) * in_rate + max(0, output_tokens or 0) * out_rate) / 1_000_000
    except (KeyError, TypeError, ValueError):
        return None

def check_budget(provider_id, model_id, estimated_input_tokens, estimated_output_tokens, config=None):
    config = config or get_spend_config()
    estimated = estimate_configured_cost(provider_id, model_id, estimated_input_tokens, estimated_output_tokens, config)
    totals = spend_totals()
    provider_totals = spend_totals(provider_id)
    if estimated is None:
        return {"allowed": True, "estimated_request_cost_usd": None, "reason": "Pricing is not configured for this exact provider/model; budget enforcement cannot estimate this request. Configure pricing to enable a preflight cap."}
    daily = config.get("daily_limit_usd")
    monthly = config.get("monthly_limit_usd")
    per_provider = config.get("provider_limits_usd", {}).get(provider_id)
    if daily is not None and totals["today_estimated_cost_usd"] + estimated > float(daily):
        return {"allowed": False, "estimated_request_cost_usd": estimated, "reason": "Daily estimated spending limit would be exceeded.", "totals": totals}
    if monthly is not None and totals["month_estimated_cost_usd"] + estimated > float(monthly):
        return {"allowed": False, "estimated_request_cost_usd": estimated, "reason": "Monthly estimated spending limit would be exceeded.", "totals": totals}
    if per_provider is not None and provider_totals["month_estimated_cost_usd"] + estimated > float(per_provider):
        return {"allowed": False, "estimated_request_cost_usd": estimated, "reason": f"Monthly spending limit for provider {provider_id} would be exceeded.", "totals": provider_totals}
    return {"allowed": True, "estimated_request_cost_usd": estimated, "reason": "Preflight check passed using configured pricing and estimated token counts.", "totals": totals}

def record_usage(provider_id, model_id, task, usage, estimated_cost_usd=None, estimated_output_tokens=None):
    con = connect()
    try:
        source = getattr(usage, "source", "unknown")
        inp = getattr(usage, "input_tokens", None)
        out = getattr(usage, "output_tokens", None)
        est_in = getattr(usage, "estimated_input_tokens", None)
        if inp is None and source != "provider-reported": inp = est_in
        if out is None and source != "provider-reported": out = estimated_output_tokens
        total = getattr(usage, "total_tokens", None)
        if total is None and source != "provider-reported" and (inp is not None or out is not None):
            total = (inp or 0) + (out or 0)
        con.execute("INSERT INTO usage_events(ts,provider_id,model_id,task,input_tokens,output_tokens,total_tokens,usage_source,estimated_input_tokens,estimated_cost_usd) VALUES(?,?,?,?,?,?,?,?,?,?)", (
            datetime.now(timezone.utc).isoformat(), provider_id, model_id, task,
            inp, out, total, source, est_in, estimated_cost_usd))
        con.commit()
    finally: con.close()

def dashboard_snapshot(providers, key_lookup):
    con = connect()
    try:
        rows = [dict(r) for r in con.execute("SELECT * FROM usage_events ORDER BY id DESC LIMIT 500")]
        totals = con.execute("SELECT COUNT(*) requests, COALESCE(SUM(total_tokens),0) tokens, COALESCE(SUM(estimated_cost_usd),0) estimated_cost FROM usage_events").fetchone()
        day = con.execute("SELECT COUNT(*) requests, COALESCE(SUM(total_tokens),0) tokens, COALESCE(SUM(estimated_cost_usd),0) estimated_cost FROM usage_events WHERE ts >= ?", (datetime.now(timezone.utc).date().isoformat(),)).fetchone()
        by_provider = {r["provider_id"]: dict(r) for r in con.execute("SELECT provider_id, COUNT(*) requests, COALESCE(SUM(total_tokens),0) tokens, COALESCE(SUM(estimated_cost_usd),0) estimated_cost, SUM(CASE WHEN usage_source='provider-reported' THEN 1 ELSE 0 END) reported_count FROM usage_events GROUP BY provider_id")}
        calibration = con.execute("SELECT COUNT(*) n, SUM(CASE WHEN total_tokens IS NOT NULL AND usage_source='provider-reported' THEN 1 ELSE 0 END) reported FROM usage_events").fetchone()
    finally: con.close()
    provider_data = []
    for p in providers:
        stat = by_provider.get(p.provider_id, {})
        key_present = bool(key_lookup(p.provider_id))
        provider_data.append({"provider_id":p.provider_id,"display_name":p.display_name,"key_configured":key_present,
          "requests":stat.get("requests",0),"tokens":stat.get("tokens",0),"estimated_cost_usd":round(stat.get("estimated_cost",0),6),
          "reported_usage_requests":stat.get("reported_count",0),"quota_status":"not exposed by current adapter",
          "models":[{"model_id":m.model_id,"display_name":m.display_name,"context_window":m.context_window,
             "capabilities":m.capabilities,"metadata_source":getattr(m,"metadata_source","starter catalog"),
             "pricing_status":"not configured"} for m in p.list_models()]})
    return {"generated_at":datetime.now(timezone.utc).isoformat(),"database":"local SQLite",
      "summary":{"requests":totals["requests"],"tokens":totals["tokens"],"estimated_cost_usd":round(totals["estimated_cost"],6),
      "today_requests":day["requests"],"today_tokens":day["tokens"],"today_estimated_cost_usd":round(day["estimated_cost"],6)},
      "calibration":{"total_requests":calibration["n"],"provider_reported_requests":calibration["reported"] or 0,
        "note":"Estimates improve only where provider-reported token usage is available. Current adapters use provider-reported counts when supplied; otherwise token estimates remain approximate."},
      "providers":provider_data,"recent_events":rows}

def estimate_input_tokens(text: str) -> int:
    """Roughly estimate input tokens; this is not provider-reported usage."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)