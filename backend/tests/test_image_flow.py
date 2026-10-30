"""Image generation: the app tries the model, records what happened, and tells the user plainly."""
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from _stubs import install
install()

os.environ["LLMHUB_TELEMETRY_DB"] = os.path.join(tempfile.mkdtemp(), "flow.sqlite3")

import main  # noqa: E402
from llm_hub import model_registry as reg  # noqa: E402
from llm_hub.providers import gemini_provider  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake"
USAGE = NS(prompt_token_count=3, candidates_token_count=4, total_token_count=7)


def reply(*parts):
    return NS(candidates=[NS(content=NS(parts=list(parts)))], text="", usage_metadata=USAGE)


TEXT = lambda t: NS(text=t, inline_data=None)  # noqa: E731
IMAGE = NS(text=None, inline_data=NS(data=PNG, mime_type="image/png"))


class FakeGemini:
    """Stands in for genai.Client; `outcome` is a response object or an exception to raise."""
    outcome = None
    last_config = None

    def __init__(self, api_key):
        outer = self

        class Models:
            def generate_content(self, model, contents, config):
                if isinstance(FakeGemini.outcome, Exception):
                    raise FakeGemini.outcome
                return FakeGemini.outcome

        self.models = Models()


def request(model, task="Image generation", provider="gemini", **kw):
    return main.ChatRequest(task=task, provider_id=provider, model_id=model,
                            messages=[{"role": "user", "content": "A cute fluffy cat playing with yarn."}], **kw)


class ImageFlow(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch.object(main, "get_provider_key", lambda pid: "key"),
                   mock.patch.object(main, "get_catalog", lambda *a, **k: []),
                   mock.patch.object(gemini_provider.genai, "Client", FakeGemini)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def check(self, model):
        return reg.get_capability_checks().get(("gemini", model, "image_generation"))

    def test_image_is_returned_and_remembered(self):
        FakeGemini.outcome = reply(TEXT("Here is a cat"), IMAGE)
        out = main.chat(request("flow-ok-image"))
        self.assertEqual(len(out["images"]), 1)
        self.assertTrue(self.check("flow-ok-image")["result"])

    def test_text_only_reply_is_called_out_and_blocks_repeat_requests(self):
        FakeGemini.outcome = reply(TEXT("Here is a cute fluffy cat!"))  # the model claims success, no image
        out = main.chat(request("flow-text-only"))
        self.assertTrue(out["answer"].startswith("**This model can't generate images.**"))
        self.assertIn("capability_notice", out)
        self.assertFalse(self.check("flow-text-only")["result"])
        with self.assertRaises(main.HTTPException) as ctx:
            main.chat(request("flow-text-only"))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("was tested on", ctx.exception.detail)

    def test_provider_rejecting_image_output_is_recorded(self):
        FakeGemini.outcome = Exception("400 INVALID_ARGUMENT: the requested response modalities are not supported")
        with self.assertRaises(main.HTTPException) as ctx:
            main.chat(request("flow-rejected"))
        self.assertIn("can't generate images", ctx.exception.detail)
        self.assertFalse(self.check("flow-rejected")["result"])

    def test_unrelated_errors_are_not_recorded_as_incapable(self):
        FakeGemini.outcome = Exception("401 API key not valid")
        with self.assertRaises(main.HTTPException) as ctx:
            main.chat(request("flow-auth-error"))
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIsNone(self.check("flow-auth-error"))

    def test_empty_reply_is_inconclusive(self):
        FakeGemini.outcome = reply()
        out = main.chat(request("flow-blocked"))
        self.assertTrue(out["answer"].startswith("The model returned no content"))
        self.assertIsNone(self.check("flow-blocked"))

    def test_provider_without_an_image_adapter_gets_an_honest_reason(self):
        with self.assertRaises(main.HTTPException) as ctx:
            main.chat(request("gpt-4.1", provider="openai"))
        self.assertIn("can't be used for image generation in this app yet", ctx.exception.detail)
        self.assertIn("limit of the app", ctx.exception.detail)

    def test_test_endpoint_verifies_and_can_retest_after_a_failure(self):
        FakeGemini.outcome = reply(TEXT("sorry, text only"))
        r = main.test_image_output(main.ImageTestRequest(provider_id="gemini", model_id="flow-retest"))
        self.assertEqual(r["status"], "text_only")
        FakeGemini.outcome = reply(IMAGE)
        r = main.test_image_output(main.ImageTestRequest(provider_id="gemini", model_id="flow-retest"))
        self.assertEqual(r["status"], "verified")
        self.assertTrue(self.check("flow-retest")["result"])

    def test_auto_picks_a_model_that_has_produced_images(self):
        reg.add_custom_model("gemini", "flow-auto-image", "Auto image", ["text"], None, "paid")
        reg.record_capability_check("gemini", "flow-auto-image", "image_generation", True, "Returned an image")
        FakeGemini.outcome = reply(IMAGE)
        out = main.chat(request(None, provider=None, auto=True))
        self.assertEqual(out["model_id"], "flow-auto-image")

    def test_auto_without_any_known_image_model_explains_what_to_do(self):
        with mock.patch.object(reg, "get_capability_checks", return_value={}), \
             mock.patch.object(reg, "list_custom_models", return_value=[]):
            with self.assertRaises(main.HTTPException) as ctx:
                main.chat(request(None, provider=None, auto=True))
        self.assertIn("will try it and remember the result", ctx.exception.detail)


class CatalogStatus(unittest.TestCase):
    def test_status_labels(self):
        checks = {("gemini", "a", "image_generation"): {"result": True, "evidence": "Returned an image", "checked_at": "2026-10-22T10:00:00"},
                  ("gemini", "b", "image_generation"): {"result": False, "evidence": "Replied with text only", "checked_at": "2026-10-22T10:00:00"}}
        self.assertEqual(reg.image_output_status("gemini", "a", [], checks)["status"], "verified")
        self.assertEqual(reg.image_output_status("gemini", "b", ["image_generation"], checks)["status"], "failed")
        self.assertEqual(reg.image_output_status("gemini", "c", ["image_generation"], checks)["status"], "declared")
        self.assertEqual(reg.image_output_status("gemini", "d", [], checks)["status"], "untested")
        self.assertEqual(reg.image_output_status("openai", "gpt", [], checks)["status"], "unsupported_adapter")

    def test_a_failed_test_removes_a_declared_capability_and_a_pass_adds_it(self):
        checks = {("gemini", "a", "image_generation"): {"result": True, "evidence": "", "checked_at": "x"},
                  ("gemini", "b", "image_generation"): {"result": False, "evidence": "", "checked_at": "x"}}
        self.assertIn("image_generation", reg._apply_checks(["text"], "gemini", "a", checks))
        self.assertNotIn("image_generation", reg._apply_checks(["text", "image_generation"], "gemini", "b", checks))


if __name__ == "__main__":
    unittest.main()
