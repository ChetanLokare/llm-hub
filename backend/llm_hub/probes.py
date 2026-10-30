"""Model Lab: send small, different requests to a model and record what it can really do.

Every check is a real request with a verifiable answer where one is possible:
  text        "reply pong"                       -> works / doesn't
  code        write add(a, b)                    -> answer contains a function
  image       a solid-colour PNG                 -> must name the colour (a model that ignores the image fails)
  pdf         a PDF containing a secret code     -> must repeat the code
  audio       a 1-second tone (WAV)              -> accepted (what it hears can't be verified)
  video       a 1-second clip made with ffmpeg   -> accepted (needs ffmpeg installed, or upload a sample)
Image OUTPUT is checked by the app's existing image test (see main.py).

Results are written to the same capability memory that Auto-pick and Safe fallback read. An inconclusive result
(rate limit, server error, timeout) is reported but NOT recorded, so a bad moment never counts against a model.
This module imports no SDKs.
"""
from __future__ import annotations

import io
import math
import random
import shutil
import struct
import subprocess
import tempfile
import time
import wave
import zlib
from dataclasses import dataclass, field
from typing import Callable

from .attachments import Part, Prepared, render_user_content
from .capabilities import CHECK_PREFIX, failure_kind, short_error

TEXT_KINDS = ("text", "code")
INPUT_KINDS = ("image", "pdf", "audio", "video")
ALL_KINDS = TEXT_KINDS + INPUT_KINDS + ("image_output",)
WORKS_PREFIX = "works:"

_COLOURS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 70, 220), "yellow": (240, 210, 20)}


# ----------------------------------------------------------------------------- test assets
def make_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    row = b"\x00" + bytes(rgb) * size
    raw = zlib.compress(row * size)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def make_pdf(text: str) -> bytes:
    """A minimal valid one-page PDF with `text` in Helvetica."""
    stream = f"BT /F1 20 Tf 50 700 Td ({text}) Tj ET".encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_wav(seconds: float = 1.0, freq: int = 440, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
                               for i in range(int(rate * seconds))))
    return buf.getvalue()


def make_mp4() -> bytes | None:
    """A tiny clip, only if ffmpeg is installed. No bundled video: returns None otherwise."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/clip.mp4"
        try:
            subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1:r=10", "-pix_fmt", "yuv420p", out],
                           check=True, timeout=30, capture_output=True)
            return open(out, "rb").read()
        except Exception:
            return None


# ----------------------------------------------------------------------------- probes
@dataclass
class ProbeResult:
    kind: str
    status: str            # pass | fail | inconclusive | skipped
    detail: str
    ms: int | None = None
    recorded: bool = False

    def as_dict(self) -> dict:
        return {"kind": self.kind, "status": self.status, "detail": self.detail, "ms": self.ms, "recorded": self.recorded}


@dataclass
class Case:
    kind: str
    prompt: str
    prepared: Prepared = field(default_factory=Prepared)
    verify: Callable[[str], bool] | None = None
    capability: str = ""


def build_cases(kinds: list[str], *, video_sample: bytes | None = None, rng: random.Random | None = None) -> tuple[list[Case], list[ProbeResult]]:
    rng = rng or random.Random()
    cases: list[Case] = []
    skipped: list[ProbeResult] = []
    for kind in kinds:
        if kind == "text":
            cases.append(Case("text", "Reply with exactly one word: pong", verify=lambda a: "pong" in a.lower(),
                              capability=WORKS_PREFIX + "text"))
        elif kind == "code":
            cases.append(Case("code", "Write a Python function add(a, b) that returns a + b. Reply with only the code.",
                              verify=lambda a: "def add" in a and "return" in a, capability=WORKS_PREFIX + "code"))
        elif kind == "image":
            name = rng.choice(list(_COLOURS))
            png = make_png(_COLOURS[name])
            cases.append(Case("image", "What is the main colour of the attached image? Answer with one word.",
                              Prepared([Part("probe.png", "image", "image/png", png)]),
                              verify=lambda a, n=name: n in a.lower(), capability=CHECK_PREFIX + "image"))
        elif kind == "pdf":
            code = "PINEAPPLE-" + str(rng.randint(100, 999))
            pdf = make_pdf(f"The secret code is {code}.")
            cases.append(Case("pdf", "What is the secret code in the attached PDF? Reply with only the code.",
                              Prepared([Part("probe.pdf", "pdf", "application/pdf", pdf, None)]),  # text=None forces native
                              verify=lambda a, c=code: c.lower() in a.lower(), capability=CHECK_PREFIX + "pdf"))
        elif kind == "audio":
            cases.append(Case("audio", "Is there an audible tone in the attached audio? Answer yes or no.",
                              Prepared([Part("probe.wav", "audio", "audio/wav", make_wav())]), verify=None,
                              capability=CHECK_PREFIX + "audio"))
        elif kind == "video":
            clip = video_sample or make_mp4()
            if clip is None:
                skipped.append(ProbeResult("video", "skipped", "No sample video: install ffmpeg or upload a short clip."))
            else:
                cases.append(Case("video", "Briefly describe the attached video in one sentence.",
                                  Prepared([Part("probe.mp4", "video", "video/mp4", clip)]), verify=None,
                                  capability=CHECK_PREFIX + "video"))
    return cases, skipped


def run_case(case: Case, send: Callable[[list[dict]], str], record: Callable, clock: Callable[[], float] = time.monotonic) -> ProbeResult:
    """Run one probe. `send(messages) -> answer text`; raises on any provider error."""
    content = render_user_content(case.prompt, case.prepared, lambda kind: True)
    started = clock()
    try:
        answer = (send([{"role": "user", "content": content}]) or "").strip()
    except Exception as exc:  # noqa: BLE001
        ms = int((clock() - started) * 1000)
        kind = failure_kind(exc)
        detail = short_error(exc)
        if kind == "modality" and case.kind in INPUT_KINDS:
            record(case.capability, False, f"Rejected {case.kind} input: {detail}")
            return ProbeResult(case.kind, "fail", f"Rejected this input: {detail}", ms, True)
        if kind == "not_found":
            record(WORKS_PREFIX + "text", False, f"Model not available: {detail}")
            return ProbeResult(case.kind, "fail", f"Model not available: {detail}", ms, True)
        label = {"rate_limit": "Rate-limited", "auth": "Key rejected", "server": "Provider error / timeout"}.get(kind, "Error")
        return ProbeResult(case.kind, "inconclusive", f"{label}: {detail}", ms, False)
    ms = int((clock() - started) * 1000)
    if not answer:
        return ProbeResult(case.kind, "inconclusive", "Empty reply (possibly a safety filter); nothing recorded.", ms)

    verified = case.verify(answer) if case.verify else None
    if case.kind in TEXT_KINDS:
        if case.kind == "text":
            record(case.capability, True, "Answered" + (" correctly" if verified else " (not the expected word)"))
        else:
            record(case.capability, bool(verified), "Wrote a function" if verified else "Answered, but no function in the reply")
        ok = True if case.kind == "text" else bool(verified)
        return ProbeResult(case.kind, "pass" if ok else "fail", "Correct answer." if ok else f"Unexpected answer: {answer[:80]}", ms, True)
    if verified is False:  # answered, but did not use the file: does NOT count as accepting the input
        record(case.capability, False, f"Answered but did not use the {case.kind}: {answer[:60]}")
        return ProbeResult(case.kind, "fail", f"Replied without using the {case.kind}: {answer[:80]}", ms, True)
    record(case.capability, True, "Verified from its answer" if verified else "Accepted (answer can't be verified)")
    return ProbeResult(case.kind, "pass", "Read it correctly." if verified else "Accepted it.", ms, True)


def run_probe(kinds: list[str], send_with: Callable[[Prepared, list[dict]], str], record_for: Callable[[str, bool, str], None], *,
              video_sample: bytes | None = None, rng: random.Random | None = None) -> list[ProbeResult]:
    """Run all requested (non-output) probes in a sensible order. Stops early if the key is rejected or the model is missing."""
    ordered = [k for k in ALL_KINDS if k in kinds and k != "image_output"]
    cases, results = build_cases(ordered, video_sample=video_sample, rng=rng)
    for case in cases:
        res = run_case(case, lambda messages, c=case: send_with(c.prepared, messages), record_for)
        results.append(res)
        if res.status == "inconclusive" and ("Key rejected" in res.detail):
            break
        if res.status == "fail" and "Model not available" in res.detail:
            break
    order = {k: i for i, k in enumerate(ALL_KINDS)}
    return sorted(results, key=lambda r: order.get(r.kind, 99))
