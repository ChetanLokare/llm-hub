import base64
import unittest

from _stubs import install
install()

from llm_hub.capabilities import UnsupportedModality  # noqa: E402
from llm_hub.providers.parts import to_anthropic_content, to_gemini_items, to_openai_content  # noqa: E402

D = base64.b64encode(b"bytes").decode()


def media(kind, mime, name="f"): return {"type": "media", "kind": kind, "mime_type": mime, "name": name, "data": D}


class Converters(unittest.TestCase):
    def test_strings_pass_through(self):
        self.assertEqual(to_openai_content("hi"), "hi")
        self.assertEqual(to_anthropic_content("hi"), "hi")

    def test_openai(self):
        out = to_openai_content([{"type": "text", "text": "q"}, media("image", "image/png"), media("pdf", "application/pdf", "a.pdf"),
                                 media("audio", "audio/wav")])
        self.assertEqual([p["type"] for p in out], ["text", "image_url", "file", "input_audio"])
        self.assertTrue(out[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        with self.assertRaises(UnsupportedModality):
            to_openai_content([media("video", "video/mp4")])
        self.assertEqual(to_openai_content([media("video", "video/mp4")], video_part=True)[0]["type"], "video_url")
        with self.assertRaises(UnsupportedModality):
            to_openai_content([media("audio", "audio/ogg")])

    def test_anthropic(self):
        out = to_anthropic_content([media("image", "image/jpeg"), media("pdf", "application/pdf", "x.pdf")])
        self.assertEqual([p["type"] for p in out], ["image", "document"])
        for kind, mime in (("video", "video/mp4"), ("audio", "audio/wav")):
            with self.assertRaises(UnsupportedModality):
                to_anthropic_content([media(kind, mime)])

    def test_gemini_accepts_everything_inline(self):
        items = to_gemini_items([{"type": "text", "text": "q"}, media("video", "video/mp4"), media("audio", "audio/ogg")])
        self.assertEqual([i[0] for i in items], ["text", "bytes", "bytes"])
        self.assertEqual(items[1][2], b"bytes")


if __name__ == "__main__":
    unittest.main()
