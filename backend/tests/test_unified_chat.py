"""One chat window: local memory across model switches, edit / regenerate, intent-based image output, Model Lab."""
import base64
import os
import re
import tempfile
import unittest
import zlib
from types import SimpleNamespace as NS
from unittest import mock

from _stubs import install
install()

os.environ["LLMHUB_TELEMETRY_DB"] = os.path.join(tempfile.mkdtemp(), "unified.sqlite3")

import main  # noqa: E402
from llm_hub import conversations as conv  # noqa: E402
from llm_hub import model_registry as reg  # noqa: E402
from llm_hub import probes  # noqa: E402
from llm_hub.capabilities import HEALTH, RECENT_USE, UnsupportedModality  # noqa: E402
from llm_hub.intent import detect_output_intent  # noqa: E402
from llm_hub.models import ModelInfo, UsageInfo  # noqa: E402
from llm_hub.providers import gemini_provider  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
USAGE = UsageInfo(3, 4, 7, "provider-reported")


class Err(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status_code = status


def blocks(messages):
    c = messages[-1]["content"]
    return c if isinstance(c, list) else [{"type": "text", "text": c}]


def png_colour(b64):
    raw = base64.b64decode(b64)
    pos, idat = 8, b""
    while pos < len(raw):
        n = int.from_bytes(raw[pos:pos + 4], "big"); tag = raw[pos + 4:pos + 8]
        if tag == b"IDAT":
            idat += raw[pos + 8:pos + 8 + n]
        pos += 12 + n
    px = zlib.decompress(idat)[1:4]
    return min(probes._COLOURS, key=lambda k: sum((a - b) ** 2 for a, b in zip(probes._COLOURS[k], px)))


def seeing_model(messages):
    """Behaves like a competent multimodal model."""
    out = []
    for b in blocks(messages):
        if b["type"] == "media" and b["kind"] == "image":
            out.append(png_colour(b["data"]))
        elif b["type"] == "media" and b["kind"] == "pdf":
            out.append(re.search(rb"\((.*?)\) Tj", base64.b64decode(b["data"])).group(1).decode().split()[-1].rstrip("."))
        elif b["type"] == "media":
            out.append("yes")
        else:
            t = b["text"].lower()
            out.append("pong" if "pong" in t else "def add(a, b):\n    return a + b" if "add(a, b)" in t else "ok")
    return " ".join(out)


def blind_model(messages):
    return "I'm sorry, I can't see any attachment."


class Fake:
    def __init__(self, pid, models, behave=None):
        self.provider_id, self.display_name, self._m = pid, pid.title(), models
        self.behave, self.calls = behave or {}, []

    def list_models(self):
        return self._m

    def chat(self, model_id, messages, api_key, system_prompt=""):
        self.calls.append({"model": model_id, "messages": messages, "system": system_prompt})
        out = self.behave.get(model_id, "answer from " + self.provider_id)
        if isinstance(out, BaseException):
            raise out
        if callable(out):
            out = out(messages)
        return out, USAGE


def M(pid, mid, support=None):
    return ModelInfo(pid.title(), mid, mid, 200_000, ["text"], "t", "unknown", support)


def att(name, data):
    return main.AttachmentIn(name=name, mime_type="", data=base64.b64encode(data).decode())


def reset_db():
    from llm_hub import telemetry
    assert str(telemetry.DB_PATH).startswith(tempfile.gettempdir()), f"refusing to wipe a non-temporary database: {telemetry.DB_PATH}"
    con = reg._con()
    con.execute("DELETE FROM capability_checks"); con.commit(); con.close()
    con = conv._con()
    for t in ("chat_files", "chat_messages", "chats"):
        con.execute(f"DELETE FROM {t}")
    con.commit(); con.close()


class Base(unittest.TestCase):
    def setUp(self):
        HEALTH.clear(); RECENT_USE.clear(); reset_db()
        self.catalog = []
        self.keys = {"a": "k", "b": "k", "gemini": "k"}
        for p in (mock.patch.object(main, "get_catalog", lambda *a, **k: self.catalog),
                  mock.patch.object(main, "get_provider_key", lambda pid: self.keys.get(pid))):
            p.start(); self.addCleanup(p.stop)

    def use(self, *fakes):
        p = mock.patch.object(main, "providers", list(fakes)); p.start(); self.addCleanup(p.stop)
        return fakes


class ChatEndpoints(Base):
    def test_create_with_context_then_first_message_uses_it(self):
        a, = self.use(Fake("a", [M("a", "a1")]))
        row = main.create_chat_endpoint(main.ChatCreateRequest(id="abc123", context="Reply in French"))
        self.assertEqual(row["id"], "abc123")
        main.chat(main.ChatRequest(message="hello", chat_id="abc123", provider_id="a", model_id="a1"))
        self.assertIn("Reply in French", a.calls[0]["system"])

    def test_import_chats_saved_by_the_earlier_browser_only_version(self):
        a, = self.use(Fake("a", [M("a", "a1")]))
        res = main.import_chat(main.ChatImportRequest(title="Old chat", context="old notes", messages=[
            {"role": "user", "content": "my cat is called Miso"}, {"role": "assistant", "content": "Nice name!"},
            {"role": "system", "content": "ignored"}, {"role": "user", "content": ""}]))
        msgs = main.get_chat(res["id"])
        self.assertEqual([m["content"] for m in msgs["messages"]], ["my cat is called Miso", "Nice name!"])
        main.chat(main.ChatRequest(message="what is my cat called?", chat_id=res["id"], provider_id="a", model_id="a1"))
        self.assertIn("Miso", str(a.calls[0]["messages"][0]["content"]))    # imported history is real context
        self.assertIn("old notes", a.calls[0]["system"])


class Intent(unittest.TestCase):
    def test_images_text_and_video(self):
        for t in ("Generate an image of a red fox", "draw a logo for my cafe", "make me a poster for a jazz night",
                  "create a picture of mountains at dawn", "an image of a cat wearing a hat"):
            self.assertEqual(detect_output_intent(t), "image", t)
        for t in ("write python code that generates an image", "explain how to create a logo", "hello", "make a function that sorts",
                  "describe this picture in detail"):
            self.assertEqual(detect_output_intent(t), "text", t)
        self.assertEqual(detect_output_intent("make a video of ocean waves"), "video")
        self.assertEqual(detect_output_intent("create an image like this", has_media_attached=True), "text")


class LocalMemory(Base):
    def test_history_and_title_persist_and_reach_the_next_model(self):
        a, b = self.use(Fake("a", [M("a", "a1")]), Fake("b", [M("b", "b1")]))
        r1 = main.chat(main.ChatRequest(message="My project is called Falcon", provider_id="a", model_id="a1"))
        cid = r1["chat_id"]
        self.assertEqual(r1["title"], "My project is called Falcon")
        # the user switches model for the next turn: the new model still sees the whole conversation
        main.chat(main.ChatRequest(message="What is my project called?", chat_id=cid, provider_id="b", model_id="b1"))
        sent = b.calls[0]["messages"]
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "user"])
        self.assertIn("Falcon", sent[0]["content"])
        self.assertEqual(len(conv.messages(cid)), 4)

    def test_safe_fallback_hands_the_same_context_to_the_next_model(self):
        a, b = self.use(Fake("a", [M("a", "a1")], {"a1": Err("down", 503)}), Fake("b", [M("b", "b1")]))
        main.chat(main.ChatRequest(message="first", provider_id="b", model_id="b1"))
        cid = conv.list_chats()[0]["id"]
        b.calls.clear()
        out = main.chat(main.ChatRequest(message="second", chat_id=cid, provider_id="a", model_id="a1", safe_fallback=True))
        self.assertEqual(out["provider_id"], "b")
        self.assertEqual(b.calls[0]["messages"][0]["content"], "first")  # same history as a would have had

    def test_earlier_image_and_file_text_follow_you_to_a_different_model(self):
        a, b = self.use(Fake("a", [M("a", "a1", {"image": True})], {"a1": seeing_model}),
                        Fake("b", [M("b", "b1", {"image": True})], {"b1": seeing_model}))
        r1 = main.chat(main.ChatRequest(message="what colour is this?", provider_id="a", model_id="a1",
                                        attachments=[att("pic.png", probes.make_png(probes._COLOURS["blue"])),
                                                     att("notes.txt", b"the budget is 40k")]))
        self.assertIn("blue", r1["answer"])
        out = main.chat(main.ChatRequest(message="and what does the note say about money?", chat_id=r1["chat_id"],
                                         provider_id="b", model_id="b1"))
        sent = b.calls[0]["messages"]
        first = sent[0]["content"]
        self.assertIn("the budget is 40k", str(first))                              # file text carried over
        self.assertTrue(any(x.get("type") == "media" and x["kind"] == "image" for x in first))  # image carried over too
        self.assertEqual(out["provider_id"], "b")

    def test_old_media_beyond_the_recent_turns_is_named_not_resent(self):
        a, = self.use(Fake("a", [M("a", "a1", {"image": True})], {"a1": seeing_model}))
        cid = None
        for i in range(4):
            r = main.chat(main.ChatRequest(message=f"turn {i}", chat_id=cid, provider_id="a", model_id="a1",
                                           attachments=[att(f"p{i}.png", probes.make_png(probes._COLOURS["red"]))]))
            cid = r["chat_id"]
        sent = a.calls[-1]["messages"]
        media_turns = [i for i, m in enumerate(sent) if isinstance(m["content"], list) and any(x.get("type") == "media" for x in m["content"])]
        self.assertEqual(len(media_turns), 3)  # the newest turn + the last 2 earlier ones
        self.assertIn("Files attached earlier: p0.png", sent[0]["content"])

    def test_pinned_context_is_sent_with_every_model(self):
        a, b = self.use(Fake("a", [M("a", "a1")]), Fake("b", [M("b", "b1")]))
        r = main.chat(main.ChatRequest(message="hi", provider_id="a", model_id="a1"))
        main.update_chat(r["chat_id"], main.ChatUpdateRequest(context="I use Windows PowerShell"))
        main.chat(main.ChatRequest(message="next", chat_id=r["chat_id"], provider_id="b", model_id="b1"))
        self.assertIn("I use Windows PowerShell", b.calls[0]["system"])

    def test_edit_replaces_a_message_and_everything_after_it(self):
        a, = self.use(Fake("a", [M("a", "a1")]))
        r1 = main.chat(main.ChatRequest(message="original question", provider_id="a", model_id="a1",
                                        attachments=[att("n.txt", b"keep this file")]))
        cid = r1["chat_id"]
        main.chat(main.ChatRequest(message="follow up", chat_id=cid, provider_id="a", model_id="a1"))
        out = main.chat(main.ChatRequest(message="better question", chat_id=cid, edit_message_id=r1["user_message_id"],
                                         provider_id="a", model_id="a1"))
        msgs = conv.messages(cid)
        self.assertEqual([m["content"] for m in msgs if m["role"] == "user"], ["better question"])
        self.assertEqual(len(msgs), 2)
        self.assertEqual([f["name"] for f in msgs[0]["files"]], ["n.txt"])          # attachments kept by default
        self.assertIn("keep this file", str(a.calls[-1]["messages"][-1]["content"]))
        self.assertEqual(out["assistant_message_id"], msgs[1]["id"])

    def test_edit_can_drop_files(self):
        a, = self.use(Fake("a", [M("a", "a1")]))
        r1 = main.chat(main.ChatRequest(message="q", provider_id="a", model_id="a1", attachments=[att("n.txt", b"secret")]))
        main.chat(main.ChatRequest(message="q2", chat_id=r1["chat_id"], edit_message_id=r1["user_message_id"], keep_files=False,
                                   provider_id="a", model_id="a1"))
        self.assertNotIn("secret", str(a.calls[-1]["messages"][-1]["content"]))

    def test_regenerate_answers_the_same_message_again_with_any_model(self):
        a, b = self.use(Fake("a", [M("a", "a1")]), Fake("b", [M("b", "b1")], {"b1": "a different answer"}))
        r1 = main.chat(main.ChatRequest(message="explain", provider_id="a", model_id="a1", attachments=[att("n.txt", b"file body")]))
        out = main.chat(main.ChatRequest(chat_id=r1["chat_id"], regenerate=True, provider_id="b", model_id="b1"))
        msgs = conv.messages(r1["chat_id"])
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[0]["id"], r1["user_message_id"])                      # your message is untouched
        self.assertEqual(msgs[1]["content"], "a different answer")
        self.assertIn("file body", str(b.calls[0]["messages"][-1]["content"]))      # its attachment came along
        self.assertEqual(out["model_id"], "b1")

    def test_failed_send_keeps_your_message_so_you_can_retry(self):
        a, = self.use(Fake("a", [M("a", "a1")], {"a1": Err("boom", 500)}))
        with self.assertRaises(main.HTTPException):
            main.chat(main.ChatRequest(message="important", provider_id="a", model_id="a1"))
        cid = conv.list_chats()[0]["id"]
        self.assertEqual([m["role"] for m in conv.messages(cid)], ["user"])
        a.behave["a1"] = "worked"
        out = main.chat(main.ChatRequest(chat_id=cid, regenerate=True, provider_id="a", model_id="a1"))
        self.assertEqual(out["answer"], "worked")

    def test_chat_management(self):
        self.use(Fake("a", [M("a", "a1")]))
        r = main.chat(main.ChatRequest(message="searchable needle text", provider_id="a", model_id="a1"))
        self.assertEqual([c["id"] for c in main.list_chats("needle")], [r["chat_id"]])
        self.assertEqual(main.list_chats("zzzz-none"), [])
        main.update_chat(r["chat_id"], main.ChatUpdateRequest(title="Renamed"))
        self.assertEqual(main.get_chat(r["chat_id"])["title"], "Renamed")
        fid = main.get_chat(r["chat_id"])["messages"][0]
        main.delete_chat(r["chat_id"])
        with self.assertRaises(main.HTTPException):
            main.get_chat(r["chat_id"])

    def test_stored_file_download(self):
        self.use(Fake("a", [M("a", "a1")]))
        r = main.chat(main.ChatRequest(message="see", provider_id="a", model_id="a1", attachments=[att("pic.png", PNG)]))
        fid = main.get_chat(r["chat_id"])["messages"][0]["files"][0]["id"]
        resp = main.get_chat_file(r["chat_id"], fid)
        self.assertEqual(resp.body, PNG); self.assertEqual(resp.media_type, "image/png")
        with self.assertRaises(main.HTTPException):
            main.get_chat_file(r["chat_id"], 99999)

    def test_legacy_stateless_messages_still_work(self):
        a, = self.use(Fake("a", [M("a", "a1")]))
        out = main.chat(main.ChatRequest(messages=[{"role": "user", "content": "hi"}], provider_id="a", model_id="a1"))
        self.assertTrue(out["answer"]); self.assertNotIn("chat_id", out)


class OneWindowImages(Base):
    def setUp(self):
        super().setUp()
        self.outcomes = {}

        class Client:
            def __init__(inner, api_key):
                outer = self

                class Models:
                    def generate_content(_s, model, contents, config):
                        o = outer.outcomes.get(model, outer.outcomes.get("*"))
                        if isinstance(o, Exception):
                            raise o
                        return o
                inner.models = Models()
        p = mock.patch.object(gemini_provider.genai, "Client", Client); p.start(); self.addCleanup(p.stop)
        self.usage = NS(prompt_token_count=1, candidates_token_count=1, total_token_count=2)

    def reply(self, *parts):
        return NS(candidates=[NS(content=NS(parts=list(parts)))], text="", usage_metadata=self.usage)

    IMG = NS(text=None, inline_data=NS(data=b"\x89PNGfake", mime_type="image/png"))
    TXT = staticmethod(lambda t: NS(text=t, inline_data=None))

    def test_the_same_window_makes_an_image_when_you_ask_for_one(self):
        self.catalog = [{"provider_id": "gemini", "models": [{"model_id": "gemini-x-image"}]}]
        self.outcomes = {"*": self.reply(self.TXT("no image here")), "gemini-x-image": self.reply(self.TXT("here"), self.IMG)}
        self.use(*main.providers)
        out = main.chat(main.ChatRequest(message="Generate an image of a red fox", auto=True, safe_fallback=True))
        self.assertEqual(len(out["images"]), 1)
        self.assertEqual(out["output_mode"], "image")
        saved = conv.messages(out["chat_id"])[-1]
        self.assertEqual(len(saved["images"]), 1)                                   # kept in local history
        self.assertTrue(reg.get_capability_checks()[("gemini", "gemini-x-image", "image_generation")]["result"])

    def test_safe_fallback_walks_image_candidates_until_one_draws(self):
        self.catalog = [{"provider_id": "gemini", "models": [{"model_id": "gemini-m1"}, {"model_id": "gemini-m2"}, {"model_id": "gemini-winner"}]}]
        self.outcomes = {"*": self.reply(self.TXT("only words")), "gemini-winner": self.reply(self.IMG),
                         "gemini-m1": Err("400 INVALID_ARGUMENT: response modalities are not supported")}
        out = main.chat(main.ChatRequest(message="draw a logo for my cafe", auto=True, safe_fallback=True))
        self.assertEqual(len(out["images"]), 1)
        self.assertTrue(reg.get_capability_checks()[("gemini", "gemini-winner", "image_generation")]["result"])
        self.assertEqual(out["model_id"], "gemini-winner")

    def test_no_image_model_means_a_text_answer_not_an_error_when_intent_was_guessed(self):
        a, = self.use(Fake("a", [M("a", "a1")]))
        out = main.chat(main.ChatRequest(message="generate an image of a sunset", provider_id="a", model_id="a1"))
        self.assertTrue(out["answer"]); self.assertIn("couldn't", out["capability_notice"]); self.assertFalse(out.get("images"))

    def test_forcing_image_with_no_image_model_is_an_error(self):
        self.use(Fake("a", [M("a", "a1")]))
        with self.assertRaises(main.HTTPException) as cm:
            main.chat(main.ChatRequest(message="a sunset", output="image", provider_id="a", model_id="a1"))
        self.assertEqual(cm.exception.status_code, 400)

    def test_forcing_text_skips_image_intent(self):
        a, = self.use(Fake("a", [M("a", "a1")]))
        out = main.chat(main.ChatRequest(message="Generate an image of a cat", output="text", provider_id="a", model_id="a1"))
        self.assertEqual(out["output_mode"], "text"); self.assertEqual(len(a.calls), 1)


class ModelLab(Base):
    def probe(self, fake, **kw):
        self.use(fake)
        return main.probe_model(main.ProbeRequest(provider_id=fake.provider_id, model_id=fake.models_id, **kw))

    def test_a_model_that_really_sees_passes_and_is_remembered(self):
        a = Fake("a", [M("a", "good")], {"good": seeing_model}); a.models_id = "good"
        out = self.probe(a, kinds=["text", "code", "image", "pdf", "audio"])
        self.assertEqual(sorted(out["passed"]), ["audio", "code", "image", "pdf", "text"])
        checks = reg.get_capability_checks()
        for cap in ("works:text", "works:code", "input:image", "input:pdf", "input:audio"):
            self.assertTrue(checks[("a", "good", cap)]["result"], cap)
        self.assertEqual(len(a.calls), 5)  # five different requests, each small

    def test_a_model_that_ignores_the_image_does_not_get_credit(self):
        a = Fake("a", [M("a", "blind")], {"blind": blind_model}); a.models_id = "blind"
        out = self.probe(a, kinds=["text", "image"])
        by = {r["kind"]: r for r in out["results"]}
        self.assertEqual(by["text"]["status"], "pass"); self.assertEqual(by["image"]["status"], "fail")
        self.assertFalse(reg.get_capability_checks()[("a", "blind", "input:image")]["result"])

    def test_provider_rejecting_an_input_is_recorded_as_failed(self):
        a = Fake("a", [M("a", "tiny")], {"tiny": lambda m: (_ for _ in ()).throw(UnsupportedModality("no files", "pdf")) if any(
            b.get("type") == "media" for b in blocks(m)) else "pong"}); a.models_id = "tiny"
        out = self.probe(a, kinds=["text", "pdf"])
        self.assertEqual({r["kind"]: r["status"] for r in out["results"]}, {"text": "pass", "pdf": "fail"})
        self.assertFalse(reg.get_capability_checks()[("a", "tiny", "input:pdf")]["result"])

    def test_rate_limit_is_inconclusive_and_records_nothing(self):
        a = Fake("a", [M("a", "busy")], {"busy": Err("429 too many requests", 429)}); a.models_id = "busy"
        out = self.probe(a, kinds=["text", "image"])
        self.assertTrue(all(r["status"] == "inconclusive" for r in out["results"]))
        self.assertEqual({k for k in reg.get_capability_checks() if k[0] == "a"}, set())

    def test_rejected_key_stops_the_run_early(self):
        a = Fake("a", [M("a", "x")], {"x": Err("invalid api key", 401)}); a.models_id = "x"
        out = self.probe(a, kinds=["text", "code", "image", "pdf"])
        self.assertEqual(len(a.calls), 1)

    def test_missing_model_is_recorded_so_auto_pick_stops_choosing_it_first(self):
        a = Fake("a", [M("a", "gone")], {"gone": Err("The model gone does not exist", 404)}); a.models_id = "gone"
        self.probe(a, kinds=["text", "image"])
        self.assertFalse(reg.get_capability_checks()[("a", "gone", "works:text")]["result"])
        from llm_hub.auto_router import plan_attempts
        fine = Fake("b", [M("b", "fine")])
        plan = plan_attempts([a, fine], set(), set(), 0, reg.get_capability_checks(), has_key=lambda p: True)
        self.assertEqual(plan[0].provider.provider_id, "b")   # demoted, still present
        self.assertEqual(len(plan), 2)

    def test_probe_results_steer_auto_pick(self):
        good = Fake("a", [M("a", "good")], {"good": seeing_model}); good.models_id = "good"
        other = Fake("b", [M("b", "other")])
        self.use(good, other)
        main.probe_model(main.ProbeRequest(provider_id="a", model_id="good", kinds=["text", "image"]))
        for _ in range(6):
            RECENT_USE.clear()
            out = main.chat(main.ChatRequest(message="what colour?", auto=True,
                                             attachments=[att("c.png", probes.make_png(probes._COLOURS["green"]))]))
            self.assertEqual(out["provider_id"], "a")      # confirmed to see images beats an untested model

    def test_video_is_skipped_without_a_sample(self):
        a = Fake("a", [M("a", "good")], {"good": seeing_model}); a.models_id = "good"
        with mock.patch.object(probes, "make_mp4", lambda: None):
            out = self.probe(a, kinds=["text", "video"])
        self.assertEqual({r["kind"]: r["status"] for r in out["results"]}, {"text": "pass", "video": "skipped"})

    def test_video_probe_uses_an_uploaded_sample(self):
        a = Fake("a", [M("a", "good")], {"good": seeing_model}); a.models_id = "good"
        clip = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40
        out = self.probe(a, kinds=["video"], video_sample=att("clip.mp4", clip))
        self.assertEqual(out["results"][0]["status"], "pass")
        self.assertTrue(reg.get_capability_checks()[("a", "good", "input:video")]["result"])

    def test_only_providers_with_a_key_can_be_probed(self):
        a = Fake("nokey", [M("nokey", "m")]); self.use(a)
        with self.assertRaises(main.HTTPException) as cm:
            main.probe_model(main.ProbeRequest(provider_id="nokey", model_id="m"))
        self.assertEqual(cm.exception.status_code, 400)
        with self.assertRaises(main.HTTPException):
            main.probe_model(main.ProbeRequest(provider_id="unknown", model_id="m"))

    def test_image_output_probe_is_skipped_for_providers_without_an_adapter(self):
        a = Fake("a", [M("a", "good")], {"good": seeing_model}); a.models_id = "good"
        out = self.probe(a, kinds=["image_output"])
        self.assertEqual(out["results"][0]["status"], "skipped")

    def test_catalog_exposes_what_the_lab_learned(self):
        reg.record_capability_check("a", "m", "works:text", True, "Answered correctly")
        entry = reg.enrich_catalog([{"provider_id": "a", "models": [{"model_id": "m", "display_name": "m", "capabilities": ["text"]}]}])[0]
        self.assertEqual(entry["models"][0]["probes"]["text"]["status"], "yes")
        self.assertEqual(entry["models"][0]["probes"]["code"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
