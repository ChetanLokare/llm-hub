"""Turn uploaded files into something any model can be given.

* images / audio / video / PDFs are passed on as-is ("native" parts) to models that accept them
* documents (docx, xlsx, pptx, odt/ods/odp, rtf), code, logs, CSV, JSON... become plain text, so every
  text model can read them
* PDFs also get an extracted-text copy when they have a text layer, so a text-only model can still answer
* zip / tar / tar.gz / gz archives are opened IN MEMORY (nothing is written to disk, member paths are never
  used as paths) and each member is handled by the rules above

Nothing here imports a provider SDK. File type is decided from the file's bytes first, then its name.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import html
import io
import mimetypes
import re
import tarfile
import zipfile
from dataclasses import dataclass, field, replace

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 60 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 300
MAX_NATIVE_FROM_ARCHIVE = 12
MAX_TEXT_CHARS_PER_FILE = 300_000
MAX_TEXT_CHARS_TOTAL = 1_200_000
MAX_ARCHIVE_DEPTH = 2
MIN_PDF_TEXT_CHARS = 20  # below this a PDF is treated as a scan with no usable text layer

OFFICE_EXT = {"docx", "pptx", "xlsx", "xlsm", "odt", "ods", "odp"}
ARCHIVE_EXT = ("zip", "tar", "tgz", "tar.gz", "tar.bz2", "tar.xz", "gz")


class AttachmentError(ValueError):
    """The upload itself is unusable (bad base64, too large)."""


@dataclass
class Part:
    name: str
    kind: str                 # text | image | pdf | audio | video
    mime: str = ""
    data: bytes | None = None
    text: str | None = None   # extracted text; for a pdf this is the optional text-only copy


@dataclass
class Prepared:
    parts: list[Part] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def native_required(self) -> set[str]:
        """Kinds that only a model accepting that input can read (no text copy exists)."""
        return {p.kind for p in self.parts if p.kind in ("image", "audio", "video") or (p.kind == "pdf" and p.text is None)}

    def native_optional(self) -> set[str]:
        return {"pdf"} if any(p.kind == "pdf" and p.text is not None for p in self.parts) else set()

    def estimated_tokens(self) -> int:
        total = 0
        for p in self.parts:
            if p.text:
                total += len(p.text) // 4
            elif p.kind == "image":
                total += 1500
            elif p.data:
                total += len(p.data) // 750  # audio / video / scanned pdf: coarse, provider-dependent
        return total

    def names(self) -> list[str]:
        return [p.name for p in self.parts]

    def text_only(self) -> "Prepared":
        """Last-resort copy for when no model accepted the native files: keep the text, say what was dropped."""
        parts = []
        for p in self.parts:
            if p.kind == "text" or p.text is not None:
                parts.append(replace(p, kind="text", data=None, mime="text/plain"))
            else:
                parts.append(Part(p.name, "text", "text/plain", None,
                                  f"[The {p.kind} file '{p.name}' could not be read: none of the available models accepted it.]"))
        return Prepared(parts, list(self.notes))


# ----------------------------------------------------------------------------- type detection
def sniff(data: bytes, name: str, declared: str = "") -> tuple[str, str]:
    """Return (kind, mime). kind: image | pdf | audio | video | office | archive | text | binary."""
    lower = name.lower()
    ext = lower.rsplit(".", 1)[-1] if "." in lower else ""
    head = data[:16]
    if head.startswith(b"\x89PNG"):
        return "image", "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image", "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", "image/webp"
    if head.startswith(b"%PDF"):
        return "pdf", "application/pdf"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio", "audio/wav"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "video", "video/x-msvideo"
    if head.startswith(b"fLaC"):
        return "audio", "audio/flac"
    if head.startswith(b"OggS"):
        return "audio", "audio/ogg"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio", "audio/mpeg"
    if head[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "audio", "audio/mp4"
        if brand == b"qt  ":
            return "video", "video/quicktime"
        return "video", "video/mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video", "video/webm"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        if ext in OFFICE_EXT:
            return "office", ext
        return "archive", "application/zip"
    if lower.endswith(ARCHIVE_EXT) and (head.startswith(b"\x1f\x8b") or ext in ("tar", "bz2", "xz")):
        return "archive", "application/x-tar" if "tar" in lower else "application/gzip"
    if head.startswith(b"\x1f\x8b"):
        return "archive", "application/gzip"
    if ext == "rtf" and data.startswith(b"{\\rtf"):
        return "office", "rtf"
    guessed = mimetypes.guess_type(name)[0] or declared or ""
    if guessed.startswith("image/") and guessed != "image/svg+xml":
        return "image", guessed
    if guessed.startswith("audio/"):
        return "audio", guessed
    if guessed.startswith("video/"):
        return "video", guessed
    return ("text", guessed or "text/plain") if _looks_like_text(data) else ("binary", guessed or "application/octet-stream")


def _looks_like_text(data: bytes) -> bool:
    sample = data[:4096]
    if not sample:
        return True
    if b"\x00" in sample:
        return b"\xff\xfe" in sample[:2] or b"\xfe\xff" in sample[:2]  # UTF-16 BOM
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass
    printable = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b < 127 or b >= 160)
    return printable / len(sample) > 0.95


def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


# ----------------------------------------------------------------------------- text extraction
def _strip_xml(xml: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab\s*/>", "\t", xml)
    xml = re.sub(r"<w:br\s*/>", "\n", xml)
    return _strip_xml(xml).strip()


def _pptx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[0]))
        out = []
        for i, n in enumerate(slides, 1):
            xml = z.read(n).decode("utf-8", errors="replace")
            out.append(f"## Slide {i}\n" + _strip_xml(re.sub(r"</a:p>", "\n", xml)).strip())
    return "\n\n".join(out)


def _xlsx_text(data: bytes) -> str:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            out.append(f"## Sheet: {ws.title}")
            for n, row in enumerate(ws.iter_rows(values_only=True)):
                if n >= 5000:
                    out.append("[…more rows omitted]")
                    break
                if any(c is not None for c in row):
                    out.append("\t".join("" if c is None else str(c) for c in row))
        return "\n".join(out)
    except ImportError:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if "xl/sharedStrings.xml" not in z.namelist():
                return ""
            xml = z.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
        return "(cell text only; install openpyxl for full sheets)\n" + "\n".join(
            _strip_xml(m) for m in re.findall(r"<si>(.*?)</si>", xml, flags=re.S))


def _odf_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("content.xml").decode("utf-8", errors="replace")
    return _strip_xml(re.sub(r"</text:(p|h)>|</table:table-row>", "\n", xml)).strip()


def _rtf_text(data: bytes) -> str:
    text = _decode(data)
    text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: bytes.fromhex(m.group(1)).decode("cp1252", "replace"), text)
    text = re.sub(r"\\(par|line)\b", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?|[{}]", "", text)
    return text.strip()


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return ""
        pages = []
        for n, page in enumerate(reader.pages[:300], 1):
            pages.append(f"--- page {n} ---\n{(page.extract_text() or '').strip()}")
        return "\n\n".join(pages)
    except Exception:
        return ""


def extract_office_text(ext: str, data: bytes) -> str:
    if ext == "docx":
        return _docx_text(data)
    if ext == "pptx":
        return _pptx_text(data)
    if ext in ("xlsx", "xlsm"):
        return _xlsx_text(data)
    if ext in ("odt", "ods", "odp"):
        return _odf_text(data)
    if ext == "rtf":
        return _rtf_text(data)
    return ""


# ----------------------------------------------------------------------------- preparing
class _State:
    def __init__(self) -> None:
        self.total_bytes = 0
        self.text_chars = 0
        self.native_from_archive = 0


def _clip(text: str, state: _State, notes: list[str], name: str) -> str:
    room = max(0, MAX_TEXT_CHARS_TOTAL - state.text_chars)
    limit = min(MAX_TEXT_CHARS_PER_FILE, room)
    if len(text) > limit:
        notes.append(f"'{name}' is long, so only its first {limit:,} characters were included.")
        text = text[:limit]
    state.text_chars += len(text)
    return text


def _add(prep: Prepared, state: _State, name: str, declared: str, data: bytes, depth: int, in_archive: bool) -> None:
    if len(data) > MAX_FILE_BYTES:
        prep.notes.append(f"'{name}' was skipped: larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")
        return
    state.total_bytes += len(data)
    if state.total_bytes > MAX_TOTAL_BYTES:
        prep.notes.append(f"'{name}' was skipped: the upload total passed {MAX_TOTAL_BYTES // (1024 * 1024)} MB.")
        return
    kind, mime = sniff(data, name, declared)

    if kind == "archive":
        if depth >= MAX_ARCHIVE_DEPTH:
            prep.notes.append(f"'{name}' is nested too deeply and was not opened.")
            return
        _open_archive(prep, state, name, data, depth + 1)
        return
    if kind == "office":
        try:
            text = extract_office_text(mime, data)
        except Exception as exc:  # corrupt or password-protected document
            prep.notes.append(f"'{name}' could not be read ({type(exc).__name__}).")
            return
        if not text.strip():
            prep.notes.append(f"'{name}' had no readable text.")
            return
        prep.parts.append(Part(name, "text", "text/plain", None, _clip(text, state, prep.notes, name)))
        return
    if kind == "text":
        prep.parts.append(Part(name, "text", mime, None, _clip(_decode(data), state, prep.notes, name)))
        return
    if kind == "pdf":
        text = _pdf_text(data)
        has_text = len(text.strip()) >= MIN_PDF_TEXT_CHARS
        prep.parts.append(Part(name, "pdf", "application/pdf", data,
                               _clip(text, state, prep.notes, name) if has_text else None))
        return
    if kind in ("image", "audio", "video"):
        if in_archive:
            if state.native_from_archive >= MAX_NATIVE_FROM_ARCHIVE:
                prep.notes.append(f"'{name}' was listed but not sent (limit of {MAX_NATIVE_FROM_ARCHIVE} media files per archive).")
                return
            state.native_from_archive += 1
        prep.parts.append(Part(name, kind, mime, data))
        return
    prep.notes.append(f"'{name}' is a binary file type that can't be read, so it was only listed.")


def _open_archive(prep: Prepared, state: _State, name: str, data: bytes, depth: int) -> None:
    members: list[tuple[str, bytes]] = []
    listing: list[str] = []
    skipped = 0
    try:
        lower = name.lower()
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if info.is_dir() or info.filename.startswith("__MACOSX/") or info.filename.endswith(".DS_Store"):
                        continue
                    if len(members) + skipped >= MAX_ARCHIVE_MEMBERS:
                        skipped += 1
                        continue
                    listing.append(f"{info.filename} ({info.file_size:,} bytes)")
                    if info.flag_bits & 0x1:
                        prep.notes.append(f"'{info.filename}' in {name} is password-protected and was skipped.")
                        continue
                    if info.file_size > MAX_FILE_BYTES:
                        prep.notes.append(f"'{info.filename}' in {name} is too large and was skipped.")
                        continue
                    with z.open(info) as fh:
                        members.append((info.filename, fh.read(MAX_FILE_BYTES + 1)))
        elif data.startswith(b"\x1f\x8b") and not re.search(r"\.t(ar\.)?gz$|\.tar$", lower) and not _is_tar(data):
            inner = gzip.GzipFile(fileobj=io.BytesIO(data)).read(MAX_FILE_BYTES + 1)
            members.append((re.sub(r"\.gz$", "", name, flags=re.I) or "file", inner))
            listing.append(members[0][0])
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                for m in tf:
                    if not m.isfile() or m.name.endswith(".DS_Store"):
                        continue
                    if len(members) + skipped >= MAX_ARCHIVE_MEMBERS:
                        skipped += 1
                        continue
                    listing.append(f"{m.name} ({m.size:,} bytes)")
                    if m.size > MAX_FILE_BYTES:
                        prep.notes.append(f"'{m.name}' in {name} is too large and was skipped.")
                        continue
                    fh = tf.extractfile(m)
                    if fh:
                        members.append((m.name, fh.read(MAX_FILE_BYTES + 1)))
    except Exception as exc:
        prep.notes.append(f"'{name}' could not be opened ({type(exc).__name__}).")
        return

    if skipped:
        prep.notes.append(f"'{name}' has more than {MAX_ARCHIVE_MEMBERS} files; the rest were not opened.")
    shown = "\n".join(listing[:200]) + (f"\n…and {len(listing) - 200} more" if len(listing) > 200 else "")
    prep.parts.append(Part(f"{name} (file list)", "text", "text/plain", None,
                           f"Archive '{name}' contains {len(listing)} file(s):\n{shown}"))
    for member_name, blob in members:
        # The member name is only ever used as a label, never as a filesystem path.
        label = f"{name}/{member_name.lstrip('/')}".replace("\\", "/")
        _add(prep, state, label, "", blob, depth, in_archive=True)


def _is_tar(data: bytes) -> bool:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*"):
            return True
    except Exception:
        return False


def prepare_attachments(items: list[dict]) -> Prepared:
    """items: [{"name", "mime_type", "data": base64 string}]"""
    prep, state = Prepared(), _State()
    for item in items or []:
        name = str(item.get("name") or "attachment")[:200]
        raw = item.get("data") or ""
        if "," in raw[:100] and raw.lstrip().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise AttachmentError(f"'{name}' is not valid base64 data.") from exc
        if not data:
            prep.notes.append(f"'{name}' is empty and was skipped.")
            continue
        _add(prep, state, name, str(item.get("mime_type") or ""), data, 0, in_archive=False)
    return prep


# ----------------------------------------------------------------------------- rendering
def render_user_content(text: str, prepared: Prepared, native_ok) -> str | list[dict]:
    """Build the content of the user's message.

    ``native_ok(kind) -> bool`` says whether THIS model should receive files of that kind natively.
    Returns a plain string when nothing native is attached, otherwise a list of neutral blocks:
    {"type": "text", "text"} and {"type": "media", "kind", "mime_type", "name", "data": base64}.
    """
    blocks: list[dict] = [{"type": "text", "text": text}]
    for p in prepared.parts:
        if p.kind == "text":
            blocks.append({"type": "text", "text": f"[Attached file: {p.name}]\n{p.text}\n[End of {p.name}]"})
        elif p.data is not None and (p.text is None or native_ok(p.kind)):
            blocks.append({"type": "text", "text": f"[Attached {p.kind}: {p.name}]"})
            blocks.append({"type": "media", "kind": p.kind, "mime_type": p.mime, "name": p.name,
                           "data": base64.b64encode(p.data).decode("ascii")})
        elif p.text is not None:
            blocks.append({"type": "text", "text": f"[Attached {p.kind}: {p.name} - text extracted from it]\n{p.text}\n[End of {p.name}]"})
    if prepared.notes:
        blocks.append({"type": "text", "text": "[Notes about the attachments: " + " ".join(prepared.notes) + "]"})
    if all(b["type"] == "text" for b in blocks):
        return "\n\n".join(b["text"] for b in blocks)
    return blocks


def media_kinds_in(content) -> set[str]:
    return {b["kind"] for b in content if isinstance(b, dict) and b.get("type") == "media"} if isinstance(content, list) else set()
