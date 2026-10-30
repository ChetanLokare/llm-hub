"""Helpers for model responses that contain images (no SDK imports, so they are easy to test)."""
import base64

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGES = 4


def parts_to_text_and_images(parts):
    """Split response parts into (text, images). Images become [{"mime_type", "data": base64}].

    Only common raster image types are passed on; anything else is dropped rather than sent to the browser.
    """
    texts, images = [], []
    for part in parts or []:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
        inline = getattr(part, "inline_data", None)
        if not inline:
            continue
        data = getattr(inline, "data", None)
        mime = (getattr(inline, "mime_type", "") or "").lower()
        if data and mime in ALLOWED_IMAGE_TYPES and len(images) < MAX_IMAGES:
            encoded = data if isinstance(data, str) else base64.b64encode(bytes(data)).decode("ascii")
            images.append({"mime_type": mime, "data": encoded})
    return "".join(texts), images


def is_image_output_model(model_id):
    """Gemini models that can return images in a normal chat call have 'image' in the id.
    Imagen models (id contains 'imagen') use a different endpoint that this adapter does not call."""
    mid = (model_id or "").lower()
    return "image" in mid and "imagen" not in mid
