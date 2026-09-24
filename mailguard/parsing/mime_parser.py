"""MIME walking: bodies, attachments and inline images, with attachments typed by magic bytes.

walk_mime() visits every leaf part of a message and sorts it into the
plain text body, the HTML body, attachments, or inline images. Nothing here
raises on malformed structure: a part that cannot be decoded is skipped,
because broken MIME is itself a common evasion technique and must not take
the pipeline down.

WHY MAGIC BYTES AND NOT EXTENSIONS

A filename extension and a Content-Type header are both chosen by the
sender. `invoice.pdf` can be a Windows executable and `report.txt` can be a
macro-bearing Office document; the attacker picks whatever label gets the
file past a filter and past the reader. The first bytes of the file are
what the operating system and the opening application actually act on, so
those are what we record in `magic_type`. The declared type is kept too,
in `content_type`, so a mismatch between the two is visible as evidence.
"""
from __future__ import annotations

import hashlib
from email.message import Message
from typing import Optional

from mailguard.core.models import Attachment

# (signature, name). Checked in order; longer and more specific first.
MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "ole2"),   # legacy Office, MSI, .msg
    (b"PK\x03\x04", "zip"),                          # ZIP, and OOXML (docx/xlsx/pptx)
    (b"PK\x05\x06", "zip"),                          # empty ZIP
    (b"%PDF", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"\x7fELF", "elf"),
    (b"MZ", "pe"),                                   # Windows executable / DLL
    (b"Rar!\x1a\x07", "rar"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\x1f\x8b", "gzip"),
    (b"{\\rtf", "rtf"),
)


def detect_magic_type(raw: bytes) -> str:
    """Name the real file type from its leading bytes.

    Returns one of the names in MAGIC_SIGNATURES, "ooxml" for a ZIP that
    carries an Office [Content_Types].xml, "html" / "text" for readable
    text, "empty", or "unknown".
    """
    if not raw:
        return "empty"
    for signature, name in MAGIC_SIGNATURES:
        if raw.startswith(signature):
            if name == "zip" and b"[Content_Types].xml" in raw[:4096]:
                return "ooxml"
            return name
    head = raw[:512].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        return "html"
    if head.startswith(b"<svg") or b"<svg" in head[:200]:
        return "svg"
    try:
        raw[:512].decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "unknown"


def _payload(part: Message) -> bytes:
    try:
        data = part.get_payload(decode=True)
    except Exception:
        data = None
    if data is None:
        raw = part.get_payload()
        return raw.encode("utf-8", "replace") if isinstance(raw, str) else b""
    return data if isinstance(data, bytes) else str(data).encode("utf-8", "replace")


def _decode_text(part: Message) -> str:
    """Decoded text of a text/* part: declared charset, then UTF-8, then Latin-1."""
    data = _payload(part)
    charset = None
    try:
        charset = part.get_content_charset()
    except Exception:
        pass
    for candidate in (charset, "utf-8"):
        if not candidate:
            continue
        try:
            return data.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("latin-1", errors="replace")


def _make_attachment(part: Message, filename: str) -> Attachment:
    data = _payload(part)
    try:
        content_type = (part.get_content_type() or "application/octet-stream").lower()
    except Exception:
        content_type = "application/octet-stream"
    return Attachment(
        filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        magic_type=detect_magic_type(data),
        raw=data,
    )


def _filename(part: Message) -> Optional[str]:
    try:
        name = part.get_filename()
    except Exception:
        return None
    return str(name).strip() if name else None


def walk_mime(msg: Message) -> tuple[str, str, list[Attachment], list[Attachment]]:
    """Return (text_body, html_body, attachments, inline_images).

    Inline images are image parts with a Content-ID or with
    `Content-Disposition: inline`. Everything carrying a filename or
    `Content-Disposition: attachment`, and any non-text part, is an
    attachment. Every attachment and inline image gets a SHA-256.
    """
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[Attachment] = []
    inline_images: list[Attachment] = []
    try:
        parts = list(msg.walk())
    except Exception:
        parts = [msg]
    for index, part in enumerate(parts):
        try:
            if part.is_multipart():
                continue
            content_type = (part.get_content_type() or "text/plain").lower()
            disposition = (part.get_content_disposition() or "").lower()
            filename = _filename(part)
            has_cid = bool(part.get("Content-ID"))

            if content_type.startswith("image/") and disposition != "attachment" and (has_cid or disposition == "inline"):
                subtype = content_type.split("/", 1)[1]
                inline_images.append(_make_attachment(part, filename or f"inline-{index}.{subtype}"))
            elif disposition == "attachment" or filename:
                attachments.append(_make_attachment(part, filename or f"part-{index}"))
            elif content_type == "text/html":
                html_parts.append(_decode_text(part))
            elif content_type.startswith("text/"):
                text_parts.append(_decode_text(part))
            else:
                subtype = content_type.split("/", 1)[-1]
                attachments.append(_make_attachment(part, f"part-{index}.{subtype}"))
        except Exception:
            continue
    return "\n".join(text_parts), "\n".join(html_parts), attachments, inline_images
