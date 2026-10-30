import base64
import gzip
import io
import tarfile
import unittest
import zipfile

from _stubs import install
install()

from llm_hub.attachments import (  # noqa: E402
    AttachmentError, Part, Prepared, media_kinds_in, prepare_attachments, render_user_content, sniff)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 20
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 20
WAV = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 20


def b64(b): return base64.b64encode(b).decode()
def item(name, data, mime=""): return {"name": name, "mime_type": mime, "data": b64(data)}


def make_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def make_docx(text):
    xml = "<w:document><w:body>" + "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in text.split("\n")) + "</w:body></w:document>"
    return make_zip({"word/document.xml": xml, "[Content_Types].xml": "<x/>"})


class Sniffing(unittest.TestCase):
    def test_by_bytes_not_by_name(self):
        self.assertEqual(sniff(PNG, "photo.txt")[0], "image")
        self.assertEqual(sniff(MP4, "clip.bin")[0], "video")
        self.assertEqual(sniff(WAV, "x")[0], "audio")
        self.assertEqual(sniff(b"%PDF-1.7 ...", "a")[0], "pdf")
        self.assertEqual(sniff(b"hello world", "notes.md")[0], "text")
        self.assertEqual(sniff(b"\x00\x01\x02\x03" * 50, "blob.dat")[0], "binary")
        self.assertEqual(sniff(make_zip({"a": "b"}), "x.zip")[0], "archive")
        self.assertEqual(sniff(make_docx("hi"), "x.docx")[0], "office")


class Prepare(unittest.TestCase):
    def test_text_and_code_become_text(self):
        p = prepare_attachments([item("a.py", b"print('hi')\n"), item("data.csv", b"a,b\n1,2\n")])
        self.assertEqual([x.kind for x in p.parts], ["text", "text"])
        self.assertEqual(p.native_required(), set())

    def test_docx_text_is_extracted_so_any_model_can_read_it(self):
        p = prepare_attachments([item("r.docx", make_docx("Quarterly results\nRevenue grew 12%"))])
        self.assertEqual(p.parts[0].kind, "text")
        self.assertIn("Revenue grew 12%", p.parts[0].text)

    def test_media_is_native_and_required(self):
        p = prepare_attachments([item("a.png", PNG), item("b.mp4", MP4), item("c.wav", WAV)])
        self.assertEqual(p.native_required(), {"image", "video", "audio"})

    def test_pdf_without_text_layer_requires_native(self):
        p = prepare_attachments([item("scan.pdf", b"%PDF-1.4\n%%EOF")])
        self.assertEqual(p.native_required(), {"pdf"})

    def test_zip_is_opened_in_memory_and_members_classified(self):
        z = make_zip({"notes/readme.txt": "project notes", "img/logo.png": PNG, "report.docx": make_docx("Hello docx"),
                      "tool.exe": b"\x00\x01\x02" * 100})
        p = prepare_attachments([item("bundle.zip", z)])
        kinds = {x.name: x.kind for x in p.parts}
        self.assertEqual(kinds["bundle.zip/img/logo.png"], "image")
        self.assertEqual(kinds["bundle.zip/notes/readme.txt"], "text")
        self.assertIn("Hello docx", next(x.text for x in p.parts if x.name.endswith("report.docx")))
        self.assertIn("tool.exe", next(x.text for x in p.parts if "file list" in x.name))  # listed
        self.assertTrue(any("tool.exe" in n and "binary" in n for n in p.notes))
        self.assertEqual(p.native_required(), {"image"})

    def test_zip_path_tricks_are_harmless(self):
        """Member names are labels only: nothing is ever written to disk, so ../ names cannot escape."""
        from unittest import mock
        z = make_zip({"../../etc/passwd": "root:x", "/abs/file.txt": "abs"})
        with mock.patch("builtins.open", side_effect=AssertionError("filesystem touched")), \
             mock.patch("os.makedirs", side_effect=AssertionError("filesystem touched")):
            p = prepare_attachments([item("evil.zip", z)])
        texts = {x.name.split("/", 1)[1]: x.text for x in p.parts if "file list" not in x.name}
        self.assertEqual(sorted(texts.values()), ["abs", "root:x"])
        self.assertTrue(all(x.name.startswith("evil.zip/") for x in p.parts if "file list" not in x.name))

    def test_tar_gz_and_gz(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = b"inside tar"
            ti = tarfile.TarInfo("dir/a.txt"); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        p = prepare_attachments([item("x.tar.gz", buf.getvalue())])
        self.assertTrue(any(x.kind == "text" and "inside tar" in x.text for x in p.parts))
        p = prepare_attachments([item("log.txt.gz", gzip.compress(b"gz content"))])
        self.assertTrue(any("gz content" in (x.text or "") for x in p.parts))

    def test_zip_member_limits(self):
        files = {f"f{i}.txt": "x" for i in range(400)}
        p = prepare_attachments([item("many.zip", make_zip(files))])
        self.assertTrue(any("more than 300 files" in n for n in p.notes))
        imgs = {f"i{i}.png": PNG for i in range(30)}
        p = prepare_attachments([item("imgs.zip", make_zip(imgs))])
        self.assertEqual(sum(1 for x in p.parts if x.kind == "image"), 12)

    def test_bad_input(self):
        with self.assertRaises(AttachmentError):
            prepare_attachments([{"name": "a", "data": "###not base64###\u2603"}])
        p = prepare_attachments([item("empty.txt", b"")])
        self.assertEqual(p.parts, [])

    def test_corrupt_zip_is_reported_not_fatal(self):
        p = prepare_attachments([item("broken.zip", b"PK\x03\x04garbage")])
        self.assertTrue(any("could not be opened" in n for n in p.notes))


class Render(unittest.TestCase):
    def test_text_only_collapses_to_a_string(self):
        p = prepare_attachments([item("a.txt", b"hello")])
        out = render_user_content("summarise", p, lambda k: True)
        self.assertIsInstance(out, str); self.assertIn("hello", out)

    def test_media_blocks_and_pdf_choice(self):
        pdf = Part("t.pdf", "pdf", "application/pdf", b"%PDF", "extracted words here")
        p = Prepared([pdf, Part("a.png", "image", "image/png", PNG)])
        native = render_user_content("q", p, lambda k: True)
        self.assertEqual(media_kinds_in(native), {"image", "pdf"})
        texty = render_user_content("q", p, lambda k: k == "image")
        self.assertEqual(media_kinds_in(texty), {"image"})
        self.assertIn("extracted words here", str(texty))

    def test_text_only_degrade(self):
        p = Prepared([Part("a.png", "image", "image/png", PNG), Part("n.txt", "text", "text/plain", None, "keep me")])
        t = p.text_only()
        self.assertEqual(t.native_required(), set())
        self.assertIn("could not be read", t.parts[0].text)


if __name__ == "__main__":
    unittest.main()
