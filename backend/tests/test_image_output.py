import base64
import unittest
from types import SimpleNamespace as NS

from _stubs import install
install()

from llm_hub.providers.media import is_image_output_model, parts_to_text_and_images
from llm_hub.smart_router import SmartRouter

PNG = b"\x89PNG\r\n\x1a\nfake"


class Parts(unittest.TestCase):
    def test_text_and_image_are_both_kept(self):
        parts = [NS(text="Here is a cat", inline_data=None),
                 NS(text=None, inline_data=NS(data=PNG, mime_type="image/png"))]
        text, images = parts_to_text_and_images(parts)
        self.assertEqual(text, "Here is a cat")
        self.assertEqual(images, [{"mime_type": "image/png", "data": base64.b64encode(PNG).decode()}])

    def test_unsafe_or_unknown_types_are_dropped(self):
        parts = [NS(text=None, inline_data=NS(data=b"<svg onload=x>", mime_type="image/svg+xml")),
                 NS(text=None, inline_data=NS(data=b"x", mime_type="text/html"))]
        self.assertEqual(parts_to_text_and_images(parts), ("", []))

    def test_thought_parts_are_hidden_and_count_is_capped(self):
        parts = [NS(text="private reasoning", thought=True, inline_data=None)] + \
                [NS(text=None, inline_data=NS(data=PNG, mime_type="image/png")) for _ in range(9)]
        text, images = parts_to_text_and_images(parts)
        self.assertEqual(text, "")
        self.assertEqual(len(images), 4)

    def test_image_model_detection(self):
        self.assertTrue(is_image_output_model("some-gemini-image-model"))
        self.assertFalse(is_image_output_model("imagen-anything"))
        self.assertFalse(is_image_output_model("gemini-2.5-flash"))


class FakeProvider:
    provider_id, display_name = "fake", "Fake"

    def __init__(self, result):
        self._result = result

    def list_models(self):
        return []

    def chat(self, model_id, messages, key, system_prompt=""):
        return self._result


class RouterImages(unittest.TestCase):
    def run_with(self, result):
        router = SmartRouter([], lambda pid: "key")
        return router.run(FakeProvider(result), "m", [{"role": "user", "content": "draw a cat"}])

    def test_three_tuple_images_reach_the_result(self):
        r = self.run_with(("caption", object(), [{"mime_type": "image/png", "data": "AAAA"}]))
        self.assertEqual(r.answer, "caption")
        self.assertEqual(r.images[0]["data"], "AAAA")

    def test_two_tuple_adapters_still_work(self):
        r = self.run_with(("plain text", object()))
        self.assertEqual((r.answer, r.images), ("plain text", []))


if __name__ == "__main__":
    unittest.main()
