"""End-to-end: main.chat() with attachments, Auto-pick and Safe fallback, against fake providers."""
import base64
import io
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace as NS
from unittest import mock

from _stubs import install
install()

os.environ["LLMHUB_TELEMETRY_DB"] = os.path.join(tempfile.mkdtemp(), "chatflow.sqlite3")

import main  # noqa: E402
from llm_hub import model_registry as reg  # noqa: E402
from llm_hub.capabilities import HEALTH, RECENT_USE, UnsupportedModality  # noqa: E402
from llm_hub.models import ModelInfo, UsageInfo  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


class Err(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status_code = status


class Fake:
    def __init__(self, pid, models, behave=None):
        self.provider_id, self.display_name, self._m, self.behave, self.calls = pid, pid.title(), models, behave or {}, []

    def list_models(self):
        return self._m

    def chat(self, model_id, messages, api_key, system_prompt=""):
        self.calls.append((model_id, messages))
        out = self.behave.get(model_id, "answer from " + self.provider_id)
        if isinstance(out, BaseException):
            raise out
        return out, UsageInfo(3, 4, 7, "provider-reported")


def M(pid, mid, billing="unknown", support=None):
    return ModelInfo(pid.title(), mid, mid, 200_000, ["text"], "t", billing, support)


def zipped(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def att(name, data):
    return main.AttachmentIn(name=name, mime_type="", data=base64.b64encode(data).decode())


def req(text="what is in these files?", **kw):
    return main.ChatRequest(messages=[{"role": "user", "content": text}], **kw)


class ChatFlow(unittest.TestCase):
    def setUp(self):
        HEALTH.clear(); RECENT_USE.clear()
        from llm_hub import telemetry
        assert str(telemetry.DB_PATH).startswith(tempfile.gettempdir()), f"refusing to wipe a non-temporary database: {telemetry.DB_PATH}"
        con = reg._con()  # each test starts with no remembered capability results
        con.execute("DELETE FROM capability_checks"); con.commit(); con.close()
        self.catalog = []
        self.keys = {"a": "k", "b": "k", "c": "k"}
        self.a = Fake("a", [M("a", "a1")])
        self.b = Fake("b", [M("b", "b1", support={"image": True})])
        self.c = Fake("c", [M("c", "c1", billing="free")])
        self.fakes = [self.a, self.b, self.c]
        for p in (mock.patch.object(main, "providers", self.fakes),
                  mock.patch.object(main, "get_catalog", lambda *a, **k: self.catalog),
                  mock.patch.object(main, "get_provider_key", lambda pid: self.keys.get(pid))):
            p.start(); self.addCleanup(p.stop)

    def calls(self):
        return [(f.provider_id, m) for f in self.fakes for m, _ in f.calls]

    def test_zip_of_mixed_files_reaches_a_model_that_can_read_everything(self):
        z = zipped({"notes.txt": "project alpha budget is 40k", "pic.png": PNG})
        out = main.chat(req(auto=True, safe_fallback=True, attachments=[att("bundle.zip", z)]))
        self.assertTrue(out["answer"].startswith("answer from"))
        # whoever answered received the text of the zip's file and the image as a media block
        sent = next(msgs for f in self.fakes for _, msgs in f.calls if f.provider_id == out["provider_id"])
        content = sent[-1]["content"]
        self.assertIn("project alpha budget is 40k", str(content))
        self.assertTrue(any(b.get("type") == "media" and b["kind"] == "image" for b in content))
        self.assertIn("answered_by_capable_model", out["routing_reason"].replace("Auto-selected", "answered_by_capable_model"))

    def test_image_never_goes_to_a_model_the_provider_says_cannot_see(self):
        self.a._m = [M("a", "a1", support={"image": False})]
        self.c._m = [M("c", "c1", support={"image": False})]
        for _ in range(5):
            out = main.chat(req(auto=True, attachments=[att("p.png", PNG)]))
            self.assertEqual(out["provider_id"], "b")

    def test_safe_fallback_survives_any_failure_and_says_what_happened(self):
        self.a.behave["a1"] = Err("upstream exploded", 500)
        self.b.behave["b1"] = TimeoutError("timed out")
        HEALTH.penalize("c", "c1", "rate_limit")  # make a and b go first so the fallback really happens
        out = main.chat(req(auto=True, safe_fallback=True))
        self.assertEqual(out["provider_id"], "c")
        self.assertIn("switched to", out["fallback_reason"])
        self.assertTrue(out["attempts"])

    def test_failure_without_safe_fallback_is_reported_not_hidden(self):
        self.a.behave["a1"] = Err("upstream exploded", 500)
        with self.assertRaises(main.HTTPException) as cm:
            main.chat(req(provider_id="a", model_id="a1"))
        self.assertEqual(cm.exception.status_code, 502)
        self.assertEqual(len(self.a.calls), 1); self.assertFalse(self.b.calls or self.c.calls)

    def test_when_everything_fails_the_error_lists_every_attempt(self):
        for f, m in ((self.a, "a1"), (self.b, "b1"), (self.c, "c1")):
            f.behave[m] = Err("down", 503)
        with self.assertRaises(main.HTTPException) as cm:
            main.chat(req(auto=True, safe_fallback=True))
        self.assertEqual(cm.exception.status_code, 502)
        self.assertIn("3 attempt", cm.exception.detail)

    def test_prefer_free_with_auto_picks_the_free_model_first(self):
        # "free" comes from the provider's published pricing, never from the vendor's name
        self.catalog = [{"provider_id": "c", "models": [{"model_id": "c1", "pricing": {"prompt": "0", "completion": "0"}}]}]
        for _ in range(8):
            RECENT_USE.clear()
            out = main.chat(req(auto=True, prefer_free=True))
            self.assertEqual(out["provider_id"], "c")

    def test_unreadable_attachment_still_gets_an_answer_when_fallback_is_on(self):
        for f, m in ((self.a, "a1"), (self.b, "b1"), (self.c, "c1")):
            f.behave[m] = UnsupportedModality("no images here", "image")
        # all three refuse image input; the request is re-sent as text only and a model for text answers
        def text_only_ok(messages=None):
            return "text answer"
        for f, m in ((self.a, "a1"), (self.b, "b1"), (self.c, "c1")):
            f.chat_orig = f.chat
            def chat(model_id, messages, api_key, system_prompt="", f=f):
                content = messages[-1]["content"]
                if isinstance(content, list) and any(b.get("type") == "media" for b in content):
                    raise UnsupportedModality("no images here", "image")
                f.calls.append((model_id, messages))
                return "text answer", UsageInfo(1, 1, 2, "provider-reported")
            f.chat = chat
        out = main.chat(req(auto=True, safe_fallback=True, attachments=[att("p.png", PNG), att("n.txt", b"plain notes")]))
        self.assertEqual(out["answer"], "text answer")
        self.assertTrue(out["degraded"]); self.assertEqual(out["skipped_attachments"], ["p.png"])
        # ...and that bad afternoon does not lock images out: next time the models are still tried natively
        for f in self.fakes:
            f.chat = f.chat_orig
            f.behave.clear()
        again = main.chat(req(auto=True, safe_fallback=True, attachments=[att("p.png", PNG)]))
        self.assertFalse(again["degraded"])

    def test_error_cases(self):
        with self.assertRaises(main.HTTPException) as cm:
            main.chat(req(provider_id="nope", model_id="x"))
        self.assertEqual(cm.exception.status_code, 400)
        self.keys.clear()
        with self.assertRaises(main.HTTPException) as cm:
            main.chat(req(auto=True))
        self.assertIn("API key", cm.exception.detail)
        self.keys.update(a="k")
        with self.assertRaises(main.HTTPException) as cm:
            main.chat(req(auto=True, attachments=[main.AttachmentIn(name="x", data="%%%\u2603")]))
        self.assertEqual(cm.exception.status_code, 400)

    def test_route_preview_reflects_attached_file_types(self):
        self.a._m = [M("a", "a1", support={"video": False})]
        self.b._m = [M("b", "b1", support={"video": False})]
        self.c._m = [M("c", "c1", support={"video": True})]
        prev = main.route_preview(task="General chat", prefer_free=False, modalities="video")
        self.assertEqual(prev["provider_id"], "c")
        self.assertEqual([x["provider_id"] for x in prev["alternatives"]], ["c"])


class AdapterPayloads(unittest.TestCase):
    """The real adapters, with the SDK client faked, build the right wire format."""

    def test_gemini_sends_attachments_as_bytes_parts(self):
        from llm_hub.providers import gemini_provider
        seen = {}

        class Client:
            def __init__(self, api_key):
                class Models:
                    def generate_content(_s, model, contents, config):
                        seen["contents"] = contents
                        return NS(candidates=[NS(content=NS(parts=[NS(text="ok", inline_data=None)]))], text="ok",
                                  usage_metadata=NS(prompt_token_count=1, candidates_token_count=1, total_token_count=2))
                self.models = Models()

        content = [{"type": "text", "text": "look"},
                   {"type": "media", "kind": "video", "mime_type": "video/mp4", "name": "v.mp4", "data": base64.b64encode(b"vid").decode()}]
        with mock.patch.object(gemini_provider.genai, "Client", Client):
            text, usage, images = gemini_provider.GeminiProvider().chat("m", [{"role": "user", "content": content}], "key")
        parts = seen["contents"][0].parts
        self.assertEqual(parts[0].text, "look")
        self.assertEqual(
            (parts[1].inline_data.data, parts[1].inline_data.mime_type),
            (b"vid", "video/mp4"),
        )

    def test_openai_adapter_builds_image_url_and_rejects_video(self):
        from llm_hub.providers import openai_provider
        sent = {}

        class Client:
            def __init__(self, **kw):
                outer = self
                class Completions:
                    def create(_s, model, messages):
                        sent["messages"] = messages
                        return NS(choices=[NS(message=NS(content="ok"))], usage=NS(prompt_tokens=1, completion_tokens=1, total_tokens=2))
                self.chat = NS(completions=Completions())

        img = {"type": "media", "kind": "image", "mime_type": "image/png", "name": "a.png", "data": base64.b64encode(PNG).decode()}
        with mock.patch.object(openai_provider, "OpenAI", Client):
            p = openai_provider.OpenAIProvider()
            p.chat("m", [{"role": "user", "content": [{"type": "text", "text": "q"}, img]}], "key")
            self.assertEqual(sent["messages"][0]["content"][1]["type"], "image_url")
            vid = dict(img, kind="video", mime_type="video/mp4")
            with self.assertRaises(UnsupportedModality):
                p.chat("m", [{"role": "user", "content": [vid]}], "key")


if __name__ == "__main__":
    unittest.main()
