"""Decide what kind of OUTPUT a message is asking for, so one chat window can do text and images.

This looks at the words of your message only (English), never at model names. It is a heuristic, so the chat
window shows what it decided and lets you override it (Auto / Text / Image).
"""
from __future__ import annotations

import re

_MAKE = r"(?:generate|create|make|draw|paint|render|design|produce|sketch|illustrate|imagine|give me|show me|i (?:want|need))"
_IMAGE_NOUN = (r"(?:image|images|picture|pictures|photo|photograph|illustration|logo|icon|wallpaper|poster|drawing|"
               r"artwork|portrait|painting|sketch|cartoon|avatar|banner|thumbnail|mockup|render)")
_IMAGE_REQ = re.compile(rf"\b{_MAKE}\b[^.?!\n]{{0,60}}\b{_IMAGE_NOUN}\b|\b(?:image|picture|photo|drawing|illustration) of\b"
                        rf"|^\s*/(?:image|imagine|draw)\b", re.I)
# "write code that generates an image", "explain how to create a logo" etc. are about text.
_ABOUT_TEXT = re.compile(r"\b(code|script|function|program|python|javascript|css|html|svg|prompt|explain|how (?:do|to|can)|"
                         r"describe|caption|analy[sz]e|what(?:'s| is) in)\b", re.I)
_VIDEO_REQ = re.compile(rf"\b{_MAKE}\b[^.?!\n]{{0,60}}\b(?:video|animation|animated|clip|movie|gif)\b", re.I)


def detect_output_intent(text: str, has_media_attached: bool = False) -> str:
    """Return "image", "video" or "text".

    Attached images/video mean you most likely want them *read*, so those stay text unless you force Image.
    """
    text = (text or "").strip()
    if not text:
        return "text"
    if _VIDEO_REQ.search(text):
        return "video"
    if has_media_attached:
        return "text"
    if _IMAGE_REQ.search(text) and not _ABOUT_TEXT.search(text):
        return "image"
    return "text"
