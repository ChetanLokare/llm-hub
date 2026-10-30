import unittest
from llm_hub.models import ModelInfo
from llm_hub.task_routing import choose_model


class P:
    def __init__(self, pid, models):
        self.provider_id, self.display_name, self._m = pid, pid.title(), models
    def list_models(self):
        return self._m


OA = P("openai", [ModelInfo("OpenAI", "gpt-4.1-mini", "GPT-4.1 mini", 1_000_000, ["text", "coding", "analysis"]),
                  ModelInfo("OpenAI", "gpt-4.1", "GPT-4.1", 1_000_000, ["text", "coding", "analysis"])])
AN = P("anthropic", [ModelInfo("Anthropic", "claude-3-5-haiku-latest", "Haiku", 200_000, ["text", "fast"])])
GM = P("gemini", [ModelInfo("Gemini", "gemini-2.5-flash", "Flash", 1_000_000, ["text", "vision", "coding", "analysis"])])
ALL = [OA, AN, GM]


class ChooseModel(unittest.TestCase):
    def test_manual_selection_is_honoured(self):
        s = choose_model(ALL, "General chat", "anthropic", "claude-3-5-haiku-latest")
        self.assertEqual((s.provider.provider_id, s.model_id), ("anthropic", "claude-3-5-haiku-latest"))

    def test_manual_accepts_live_catalog_model_not_in_starter_list(self):
        s = choose_model(ALL, "Coding", "openai", "some-new-live-model")
        self.assertEqual(s.model_id, "some-new-live-model")

    def test_unknown_provider_rejected(self):
        self.assertIsNone(choose_model(ALL, "Coding", "nope", "x").provider)

    def test_auto_ignores_selection_but_requires_key(self):
        s = choose_model(ALL, "Coding", "anthropic", "claude-3-5-haiku-latest", auto=True,
                         has_key=lambda pid: pid == "openai")
        self.assertEqual(s.provider.provider_id, "openai")

    def test_auto_without_any_key_explains(self):
        s = choose_model(ALL, "Coding", auto=True, has_key=lambda pid: False)
        self.assertIsNone(s.provider)
        self.assertIn("saved API key", s.reason)

    def test_media_tasks_report_unavailable_even_with_selection(self):
        s = choose_model(ALL, "Image generation", "openai", "gpt-4.1")
        self.assertIsNone(s.provider)
        self.assertIn("generate images", s.reason)

    def test_provider_without_model_uses_best_model_of_that_provider(self):
        s = choose_model(ALL, "Coding", "openai")
        self.assertEqual(s.provider.provider_id, "openai")
        self.assertIn(s.model_id, {m.model_id for m in OA.list_models()})

    def test_no_selection_prefers_providers_with_keys(self):
        s = choose_model(ALL, "Coding", has_key=lambda pid: pid == "anthropic")
        self.assertEqual(s.provider.provider_id, "anthropic")

    def test_selection_matches_what_the_ui_sent(self):
        for pid, mid in [("openai", "gpt-4.1-mini"), ("gemini", "gemini-2.5-flash"), ("anthropic", "claude-3-5-haiku-latest")]:
            s = choose_model(ALL, "General chat", pid, mid, has_key=lambda _: True)
            self.assertEqual((s.provider.provider_id, s.model_id), (pid, mid))


    def test_video_explains_that_no_adapter_exists(self):
        s = choose_model(ALL, "Video generation", "openai", "gpt-4.1")
        self.assertIsNone(s.provider)
        self.assertIn("no adapter", s.reason)


if __name__ == "__main__":
    unittest.main()
