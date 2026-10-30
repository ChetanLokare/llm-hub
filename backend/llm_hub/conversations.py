"""Local conversation memory: chats, messages and attachments, stored in the same local SQLite file as telemetry.

Why it lives here (not in the browser): the server rebuilds the context for whichever model answers next. When you
switch model, or Safe fallback switches for you, the new model still gets the same history, the text of every file you
attached, the most recent images / audio / PDFs / video you shared, and your pinned chat context.

Nothing leaves this machine except in the request to the provider you are using.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from .attachments import Part, Prepared
from .telemetry import connect

# Original file bytes are kept so an edited or regenerated message, or a different model, can re-read them.
MAX_BLOB_BYTES_PER_CHAT = 200 * 1024 * 1024
# Context policy when building a request for a model:
MAX_EARLIER_MEDIA_TURNS = 2            # last N turns that carried images / audio / video / scanned PDFs are re-sent natively
MAX_EARLIER_MEDIA_BYTES = 12 * 1024 * 1024
MAX_EARLIER_TEXT_CHARS = 120_000       # text of files attached in earlier turns, newest first; older ones are named only


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _con():
    con = connect()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS chats(
      id TEXT PRIMARY KEY, title TEXT NOT NULL, context TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS chat_messages(
      id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
      provider TEXT, model_id TEXT, note TEXT, images TEXT, meta TEXT, created_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages(chat_id, id);
    CREATE TABLE IF NOT EXISTS chat_files(
      id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL, message_id INTEGER NOT NULL, name TEXT NOT NULL,
      kind TEXT NOT NULL, mime TEXT NOT NULL, size INTEGER NOT NULL, text TEXT, data BLOB);
    CREATE INDEX IF NOT EXISTS idx_chat_files_msg ON chat_files(message_id);
    """)
    return con


class ChatNotFound(KeyError):
    pass


def create_chat(chat_id: str | None = None, title: str = "New chat") -> dict:
    chat_id = chat_id or uuid.uuid4().hex
    con = _con()
    try:
        con.execute("INSERT OR IGNORE INTO chats(id,title,context,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (chat_id, title, "", _now(), _now()))
        con.commit()
    finally:
        con.close()
    return get_chat(chat_id)


def get_chat(chat_id: str) -> dict:
    con = _con()
    try:
        row = con.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not row:
            raise ChatNotFound(chat_id)
        return dict(row)
    finally:
        con.close()


def list_chats(query: str = "", limit: int = 300) -> list[dict]:
    con = _con()
    try:
        like = f"%{query.strip()}%"
        rows = con.execute(
            """SELECT c.id, c.title, c.updated_at,
                      (SELECT COUNT(*) FROM chat_messages m WHERE m.chat_id=c.id) AS message_count
               FROM chats c
               WHERE ?='%%' OR c.title LIKE ? OR EXISTS (SELECT 1 FROM chat_messages m WHERE m.chat_id=c.id AND m.content LIKE ?)
               ORDER BY c.updated_at DESC LIMIT ?""", (like, like, like, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def update_chat(chat_id: str, title: str | None = None, context: str | None = None) -> dict:
    get_chat(chat_id)
    con = _con()
    try:
        if title is not None:
            con.execute("UPDATE chats SET title=? WHERE id=?", (title.strip()[:120] or "New chat", chat_id))
        if context is not None:
            con.execute("UPDATE chats SET context=? WHERE id=?", (context[:20000], chat_id))
        con.commit()
    finally:
        con.close()
    return get_chat(chat_id)


def delete_chat(chat_id: str) -> None:
    con = _con()
    try:
        con.execute("DELETE FROM chat_files WHERE chat_id=?", (chat_id,))
        con.execute("DELETE FROM chat_messages WHERE chat_id=?", (chat_id,))
        con.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        con.commit()
    finally:
        con.close()


def _touch(con, chat_id: str) -> None:
    con.execute("UPDATE chats SET updated_at=? WHERE id=?", (_now(), chat_id))


def add_message(chat_id: str, role: str, content: str, *, provider: str | None = None, model_id: str | None = None,
                note: str | None = None, images: list | None = None, meta: dict | None = None,
                prepared: Prepared | None = None) -> int:
    con = _con()
    try:
        cur = con.execute(
            "INSERT INTO chat_messages(chat_id,role,content,provider,model_id,note,images,meta,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (chat_id, role, content, provider, model_id, note, json.dumps(images) if images else None,
             json.dumps(meta) if meta else None, _now()))
        mid = cur.lastrowid
        for p in (prepared.parts if prepared else []):
            size = len(p.data) if p.data is not None else len((p.text or "").encode("utf-8"))
            con.execute("INSERT INTO chat_files(chat_id,message_id,name,kind,mime,size,text,data) VALUES(?,?,?,?,?,?,?,?)",
                        (chat_id, mid, p.name, p.kind, p.mime, size, p.text, p.data))
        if prepared and prepared.notes:
            con.execute("UPDATE chat_messages SET meta=? WHERE id=?",
                        (json.dumps({**(meta or {}), "attachment_notes": prepared.notes}), mid))
        _touch(con, chat_id)
        _enforce_blob_cap(con, chat_id)
        con.commit()
        return mid
    finally:
        con.close()


def _enforce_blob_cap(con, chat_id: str) -> None:
    total = con.execute("SELECT COALESCE(SUM(LENGTH(data)),0) FROM chat_files WHERE chat_id=?", (chat_id,)).fetchone()[0]
    if total <= MAX_BLOB_BYTES_PER_CHAT:
        return
    for row in con.execute("SELECT id, LENGTH(data) AS n FROM chat_files WHERE chat_id=? AND data IS NOT NULL ORDER BY id",
                           (chat_id,)).fetchall():
        con.execute("UPDATE chat_files SET data=NULL WHERE id=?", (row["id"],))
        total -= row["n"]
        if total <= MAX_BLOB_BYTES_PER_CHAT:
            break


def messages(chat_id: str, include_images: bool = True) -> list[dict]:
    get_chat(chat_id)
    con = _con()
    try:
        out = []
        files: dict[int, list] = {}
        for f in con.execute("SELECT id,message_id,name,kind,mime,size,(data IS NOT NULL) AS has_data FROM chat_files "
                             "WHERE chat_id=? ORDER BY id", (chat_id,)):
            files.setdefault(f["message_id"], []).append({"id": f["id"], "name": f["name"], "kind": f["kind"],
                                                          "mime": f["mime"], "size": f["size"], "stored": bool(f["has_data"])})
        for r in con.execute("SELECT * FROM chat_messages WHERE chat_id=? ORDER BY id", (chat_id,)):
            meta = json.loads(r["meta"]) if r["meta"] else {}
            out.append({"id": r["id"], "role": r["role"], "content": r["content"], "provider": r["provider"],
                        "model": r["model_id"], "note": r["note"], "created_at": r["created_at"],
                        "images": (json.loads(r["images"]) if (r["images"] and include_images) else None),
                        "files": files.get(r["id"], []), "meta": meta})
        return out
    finally:
        con.close()


def truncate_from(chat_id: str, message_id: int) -> None:
    """Delete this message and everything after it."""
    con = _con()
    try:
        con.execute("DELETE FROM chat_files WHERE chat_id=? AND message_id>=?", (chat_id, message_id))
        con.execute("DELETE FROM chat_messages WHERE chat_id=? AND id>=?", (chat_id, message_id))
        _touch(con, chat_id)
        con.commit()
    finally:
        con.close()


def get_message(chat_id: str, message_id: int) -> dict:
    for m in messages(chat_id, include_images=False):
        if m["id"] == message_id:
            return m
    raise ChatNotFound(f"message {message_id}")


def file_blob(chat_id: str, file_id: int) -> tuple[bytes, str, str] | None:
    con = _con()
    try:
        r = con.execute("SELECT name,mime,data FROM chat_files WHERE chat_id=? AND id=?", (chat_id, file_id)).fetchone()
        return (bytes(r["data"]), r["mime"], r["name"]) if r and r["data"] is not None else None
    finally:
        con.close()


def load_prepared(chat_id: str, message_id: int) -> Prepared:
    """Rebuild a message's attachments. A file whose bytes were dropped (size cap) becomes a plain notice."""
    con = _con()
    try:
        prep = Prepared()
        for r in con.execute("SELECT name,kind,mime,text,data FROM chat_files WHERE chat_id=? AND message_id=? ORDER BY id",
                             (chat_id, message_id)):
            data = bytes(r["data"]) if r["data"] is not None else None
            if r["kind"] in ("image", "audio", "video") and data is None:
                prep.parts.append(Part(r["name"], "text", "text/plain", None,
                                       f"[The {r['kind']} file '{r['name']}' is no longer stored on this device.]"))
            elif r["kind"] == "pdf" and data is None and r["text"] is None:
                prep.parts.append(Part(r["name"], "text", "text/plain", None,
                                       f"[The PDF '{r['name']}' is no longer stored on this device.]"))
            else:
                prep.parts.append(Part(r["name"], r["kind"], r["mime"], data, r["text"]))
        row = con.execute("SELECT meta FROM chat_messages WHERE id=?", (message_id,)).fetchone()
        if row and row["meta"]:
            prep.notes = json.loads(row["meta"]).get("attachment_notes", [])
        return prep
    finally:
        con.close()


def build_history(chat_id: str) -> tuple[list[dict], dict[int, Prepared]]:
    """The context to send to whichever model answers next.

    Returns (history, earlier_media). history is [{"role", "content"}] ending with the newest user message (its own
    attachments are supplied separately by the caller). earlier_media maps a history index to the native files of a
    recent turn, so a model that is switched in mid-conversation can still see the image you sent two messages ago.
    """
    msgs = messages(chat_id, include_images=False)
    history: list[dict] = []
    index_of: dict[int, int] = {}
    for m in msgs:
        content = m["content"]
        if m["role"] == "assistant" and m["meta"].get("had_images"):
            content = (content + "\n\n" if content else "") + "[An image was generated here.]"
        index_of[m["id"]] = len(history)
        history.append({"role": m["role"], "content": content})

    last_user_idx = max((i for i, h in enumerate(history) if h["role"] == "user"), default=-1)
    # Earlier turns: inject the TEXT of attached files (newest first, within budget) and pick recent media to re-send.
    budget = MAX_EARLIER_TEXT_CHARS
    earlier: dict[int, Prepared] = {}
    media_turns, media_bytes = 0, 0
    for m in reversed(msgs):
        if m["role"] != "user" or index_of[m["id"]] == last_user_idx or not m["files"]:
            continue
        prep = load_prepared(chat_id, m["id"])
        blocks, native = [], Prepared()
        for p in prep.parts:
            if p.kind == "text" or (p.kind == "pdf" and p.text is not None):
                label = f"[Attached file earlier in this chat: {p.name}]"
                if p.text is not None and len(p.text) <= budget:
                    blocks.append(f"{label}\n{p.text}\n[End of {p.name}]")
                    budget -= len(p.text)
                else:
                    blocks.append(f"{label} (text not repeated to save space)")
            elif p.data is not None:
                native.parts.append(p)
        i = index_of[m["id"]]
        if blocks:
            history[i]["content"] += "\n\n" + "\n\n".join(blocks)
        size = sum(len(p.data or b"") for p in native.parts)
        if native.parts and media_turns < MAX_EARLIER_MEDIA_TURNS and media_bytes + size <= MAX_EARLIER_MEDIA_BYTES:
            earlier[i] = native
            media_turns += 1
            media_bytes += size
        elif native.parts:
            history[i]["content"] += "\n[Files attached earlier: " + ", ".join(p.name for p in native.parts) + "]"
    return history, earlier
