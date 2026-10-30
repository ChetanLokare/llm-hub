import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp()
os.environ["LLMHUB_TELEMETRY_DB"] = os.path.join(_TMP, "test.sqlite3")  # must be set before importing llm_hub

from llm_hub import model_registry as reg  # noqa: E402
from llm_hub.models import ModelInfo  # noqa: E402
from llm_hub.task_routing import choose_model, rank_models  # noqa: E402


class P:
    def __init__(self, pid, models):
        self.provider_id, self.display_name, self._m = pid, pid.title(), models

    def list_models(self):
        return self._m


def M(pid, mid, ctx=128000, caps=("text", "coding", "analysis")):
    return ModelInfo(pid.title(), mid, mid, ctx, list(caps))


LINEUP = [
    P("openai", [M("openai", "gpt-4.1-mini"), M("openai", "gpt-4.1")]),
    P("anthropic", [M("anthropic", "claude-3-5-haiku-latest", 200000, ("text", "fast")),
                    M("anthropic", "claude-sonnet-4-20250514", 200000)]),
    P("gemini", [M("gemini", "gemini-2.5-flash", 1048576), M("gemini", "gemini-2.5-pro", 1048576)]),
]


class Billing(unittest.TestCase):
    def test_openrouter_zero_pricing_is_free(self):
        self.assertEqual(reg.classify_billing("openrouter", "x/y", {"prompt": "0", "completion": "0"})[0], "free")

    def test_openrouter_priced_model_is_paid(self):
        self.assertEqual(reg.classify_billing("openrouter", "x/y", {"prompt": "0.000002", "completion": "0.00001"})[0], "paid")

    def test_dynamic_pricing_is_not_guessed(self):
        self.assertEqual(reg.classify_billing("openrouter", "x/auto", {"prompt": "-1", "completion": "-1"})[0], "unknown")

    def test_free_suffix(self):
        self.assertEqual(reg.classify_billing("openrouter", "meta/llama:free")[0], "free")

    def test_no_vendor_is_assumed_free_or_paid(self):
        # Billing comes from evidence (provider pricing data, your settings), never from the vendor's name.
        for pid, mid in (("openai", "gpt-4.1"), ("anthropic", "claude-x"), ("deepseek", "deepseek-chat"), ("gemini", "gemini-2.5-flash")):
            self.assertEqual(reg.classify_billing(pid, mid)[0], "unknown", pid)

    def test_override_beats_everything(self):
        out = reg.classify_billing("openai", "gpt-4.1", overrides={("openai", "gpt-4.1"): "free"})
        self.assertEqual(out, ("free", "set by you"))

    def test_configured_zero_pricing_is_free(self):
        cfg = {"pricing": {"gemini": {"gemini-2.5-flash": {"input_per_million_usd": 0, "output_per_million_usd": 0}}}}
        self.assertEqual(reg.classify_billing("gemini", "gemini-2.5-flash", config=cfg)[0], "free")


class CustomModels(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            reg.add_custom_model("openai", "bad id with spaces")
        with self.assertRaises(ValueError):
            reg.add_custom_model("openai", "ok-model", capabilities=["telepathy"])
        with self.assertRaises(ValueError):
            reg.add_custom_model("openai", "ok-model", billing="maybe")
        with self.assertRaises(ValueError):
            reg.add_custom_model("openai", "ok-model", context_window=5)

    def test_added_model_appears_in_catalog_and_can_be_deleted(self):
        reg.add_custom_model("gemini", "gemini-9-ultra", "Gemini 9 Ultra", ["coding"], 2_000_000, "free")
        catalog = [{"provider_id": "gemini", "display_name": "Gemini", "models": [
            {"model_id": "gemini-2.5-flash", "display_name": "Flash", "capabilities": ["text"], "context_window": None}]}]
        models = {m["model_id"]: m for m in reg.enrich_catalog(catalog)[0]["models"]}
        self.assertTrue(models["gemini-9-ultra"]["custom"])
        self.assertEqual(models["gemini-9-ultra"]["billing"], "free")
        self.assertIn("text", models["gemini-9-ultra"]["capabilities"])
        self.assertFalse(models["gemini-2.5-flash"]["custom"])
        self.assertTrue(reg.delete_custom_model("gemini", "gemini-9-ultra"))
        ids = [m["model_id"] for m in reg.enrich_catalog(catalog)[0]["models"]]
        self.assertNotIn("gemini-9-ultra", ids)

    def test_added_model_is_routable_and_overrides_work(self):
        reg.add_custom_model("openai", "gpt-9-pro", "GPT-9 Pro", ["coding", "analysis"], 1_000_000, "paid")
        try:
            views = reg.routing_providers(LINEUP, [])
            ids = {m.model_id for v in views if v.provider_id == "openai" for m in v.list_models()}
            self.assertIn("gpt-9-pro", ids)
            self.assertEqual(choose_model(views, "Coding", "openai", "gpt-9-pro").model_id, "gpt-9-pro")
            reg.set_billing_override("openai", "gpt-4.1", "free")
            billing = {m.model_id: m.billing for v in reg.routing_providers(LINEUP, []) if v.provider_id == "openai"
                       for m in v.list_models()}
            self.assertEqual(billing["gpt-4.1"], "free")
            self.assertEqual(billing["gpt-4.1-mini"], "unknown")  # nothing is assumed from the vendor
        finally:
            reg.delete_custom_model("openai", "gpt-9-pro")
            reg.set_billing_override("openai", "gpt-4.1", "auto")


class AutoPick(unittest.TestCase):
    def pick(self, task, **kw):
        s = choose_model(LINEUP, task, None, None, auto=True, has_key=lambda _: True, **kw)
        return s.provider.provider_id, s.model_id

    def test_task_changes_the_pick(self):
        picks = {t: self.pick(t) for t in ("General chat", "Coding", "Research", "Writing", "Document analysis")}
        self.assertNotEqual(picks["General chat"], picks["Coding"])
        self.assertTrue(len(set(picks.values())) >= 3, picks)

    def test_dropdown_selection_does_not_decide_auto_ties(self):
        a = choose_model(LINEUP, "General chat", "openai", "gpt-4.1", auto=True, has_key=lambda _: True)
        b = choose_model(LINEUP, "General chat", "anthropic", "claude-3-5-haiku-latest", auto=True, has_key=lambda _: True)
        self.assertEqual((a.provider.provider_id, a.model_id), (b.provider.provider_id, b.model_id))

    def test_ties_go_to_the_least_used_provider(self):
        base = self.pick("General chat")
        busy = self.pick("General chat", load={base[0]: 5_000_000})
        self.assertNotEqual(base[0], busy[0])

    def test_prefer_free_picks_a_free_model(self):
        views = reg.routing_providers(LINEUP, [{"provider_id": "gemini", "models": [
            {"model_id": "gemini-2.5-flash", "pricing": {"prompt": "0", "completion": "0"}}]}])
        paid = choose_model(views, "General chat", None, None, auto=True, has_key=lambda _: True)
        free = choose_model(views, "General chat", None, None, auto=True, has_key=lambda _: True, prefer_free=True)
        self.assertEqual((free.provider.provider_id, free.model_id), ("gemini", "gemini-2.5-flash"))
        self.assertIn("free model", free.reason)
        self.assertIsNotNone(paid.model_id)

    def test_only_providers_with_keys_are_used(self):
        s = choose_model(LINEUP, "Coding", None, None, auto=True, has_key=lambda p: p == "anthropic")
        self.assertEqual(s.provider.provider_id, "anthropic")

    def test_ranking_lists_alternatives(self):
        self.assertGreaterEqual(len(rank_models(LINEUP, "Coding")), 5)


if __name__ == "__main__":
    unittest.main()
