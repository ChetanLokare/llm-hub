import unittest
from datetime import datetime, timedelta, timezone

from _stubs import install
install()

from llm_hub.capabilities import (  # noqa: E402
    redact, Health, UnsupportedModality, declared_modalities, declared_text_output, failure_kind, modality_status, short_error)


class Err(Exception):
    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status_code = status


class Declared(unittest.TestCase):
    def test_openrouter_list_is_authoritative(self):
        d = declared_modalities({"architecture": {"input_modalities": ["text", "image", "file"]}})
        self.assertEqual(d, {"image": True, "pdf": True, "audio": False, "video": False})

    def test_openrouter_modality_string(self):
        d = declared_modalities({"architecture": {"modality": "text+image->text"}})
        self.assertTrue(d["image"]); self.assertFalse(d["video"])

    def test_capability_mapping_only_answers_for_keys_it_has(self):
        d = declared_modalities({"capabilities": {"vision": True}})
        self.assertEqual(d, {"image": True})  # nothing said about pdf/audio/video -> stays unknown
        d = declared_modalities({"capabilities": {"image_input": {"supported": False}, "pdf_input": {"supported": True}}})
        self.assertEqual(d, {"image": False, "pdf": True})

    def test_silent_provider_gives_none_and_model_name_is_never_used(self):
        self.assertIsNone(declared_modalities({"id": "gpt-4o-vision-4o"}))
        self.assertIsNone(declared_modalities({"id": "x", "capabilities": ["chat"]}))

    def test_text_output(self):
        self.assertFalse(declared_text_output({"architecture": {"output_modalities": ["image"]}}))
        self.assertTrue(declared_text_output({"architecture": {"output_modalities": ["text", "image"]}}))
        self.assertFalse(declared_text_output({"type": "embedding"}))
        self.assertIsNone(declared_text_output({"id": "anything"}))


class Status(unittest.TestCase):
    def st(self, **kw):
        return modality_status("image", provider_id="p", model_id="m", **kw)[0]

    def test_unknown_by_default(self):
        self.assertEqual(self.st(), "unknown")

    def test_provider_declaration(self):
        self.assertEqual(self.st(declared={"image": True}), "yes")
        self.assertEqual(self.st(declared={"image": False}), "no")

    def test_real_result_beats_declaration(self):
        now = datetime.now(timezone.utc).isoformat()
        no = {("p", "m", "input:image"): {"result": False, "evidence": "x", "checked_at": now}}
        yes = {("p", "m", "input:image"): {"result": True, "evidence": "x", "checked_at": now}}
        self.assertEqual(self.st(declared={"image": True}, checks=no), "unlikely")
        self.assertEqual(self.st(declared={"image": False}, checks=yes), "yes")

    def test_rejections_expire(self):
        old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        no = {("p", "m", "input:image"): {"result": False, "evidence": "x", "checked_at": old}}
        self.assertEqual(self.st(checks=no), "unknown")

    def test_user_declared_vision(self):
        self.assertEqual(self.st(user_caps=["text", "vision"]), "yes")


class Failures(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(failure_kind(Err("Too Many Requests", 429)), "rate_limit")
        self.assertEqual(failure_kind(Err("RESOURCE_EXHAUSTED quota")), "rate_limit")
        self.assertEqual(failure_kind(Err("bad key", 401)), "auth")
        self.assertEqual(failure_kind(Err("The model xyz does not exist", 404)), "not_found")
        self.assertEqual(failure_kind(Err("Internal error", 500)), "server")
        self.assertEqual(failure_kind(TimeoutError("request timed out")), "server")
        self.assertEqual(failure_kind(Err("maximum context length is 8192 tokens", 400)), "too_large")
        self.assertEqual(failure_kind(UnsupportedModality("no video", "video")), "modality")
        self.assertEqual(failure_kind(Err("This model does not support image input", 400)), "modality")
        self.assertEqual(failure_kind(Err("invalid content type: file is not allowed", 400)), "modality")
        self.assertEqual(failure_kind(Err("something odd", 400)), "bad_request")
        self.assertEqual(failure_kind(Err("???")), "unknown")

    def test_redact_hides_key_like_text(self):
        for secret in ("sk-proj-abcdefghijklmnop1234", "AIzaSyA1B2C3D4E5F6G7H8I9J0", "gsk_abcdefghij12345", "Bearer abcdef0123456789xyz"):
            self.assertNotIn(secret, redact(f"Incorrect API key provided: {secret}. See docs"))
        self.assertEqual(redact("plain error"), "plain error")

    def test_secrets_are_redacted(self):
        self.assertNotIn("sk-abcdefghijkl", short_error(Err("bad key sk-abcdefghijklmnop")))


class HealthTest(unittest.TestCase):
    def test_cooldowns(self):
        t = [0.0]
        h = Health(clock=lambda: t[0])
        h.penalize("p", "m", "rate_limit")
        self.assertTrue(h.is_cooling("p", "m")); self.assertFalse(h.is_cooling("p", "other"))
        h.penalize("q", "m", "auth")
        self.assertTrue(h.is_cooling("q", "anything"))  # a bad key affects the whole provider
        t[0] = 5000
        self.assertFalse(h.is_cooling("p", "m")); self.assertFalse(h.is_cooling("q", "x"))
        h.penalize("p", "m", "modality")  # capability problems are not "cooldowns"
        self.assertFalse(h.is_cooling("p", "m"))


if __name__ == "__main__":
    unittest.main()
