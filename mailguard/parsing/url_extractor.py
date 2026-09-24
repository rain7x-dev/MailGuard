"""URL extraction from every place a link can hide, and assembly of the ParsedEmail.

Links are pulled from four places, and each ExtractedURL records which one
in its `source` field:

  "text"       a regex over the plain text body
  "html_href"  every href and src in the HTML body (stdlib html.parser)
  "qr"         QR codes decoded from inline images (pyzbar + Pillow, optional)
  "ocr"        text read out of inline images (pytesseract + Pillow, optional),
               then the same URL regex

Attackers hide links in QR codes and images precisely because most
scanners only read the text body. A phishing mail whose only link is a QR
code of a login page, or a screenshot of a URL, has no link at all as far
as a text-only filter is concerned. Reading the images closes that gap
when the optional libraries are present; when they are not, those two
sources simply contribute nothing and the rest of the pipeline is unaffected.

build_parsed_email() lives here because URL extraction is the last parsing
step: it runs the header, MIME and URL parsers and returns a fully
populated ParsedEmail with trust_boundary_index and attribution still None.
"""
from __future__ import annotations

import io
import re
from datetime import timezone
from email import message_from_bytes, policy
from email.message import Message
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlsplit

from mailguard.core.models import Attachment, ExtractedURL, ParsedEmail
from mailguard.ingest.loader import sha256_of
from mailguard.parsing.header_parser import (
    extract_addresses,
    extract_auth_results,
    extract_dkim_signatures,
    parse_headers,
    parse_message_ids,
    parse_received_chain,
)
from mailguard.parsing.mime_parser import walk_mime

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from pyzbar.pyzbar import decode as zbar_decode

    HAS_PYZBAR = True
except Exception:  # ImportError, or the zbar shared library itself is missing
    HAS_PYZBAR = False

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

SHORTENER_DOMAINS: frozenset[str] = frozenset(
    {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "rb.gy", "is.gd", "cutt.ly", "short.link"}
)

_URL_RE = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>"'`)\]]+)""")
_TRAILING = ".,;:!?)]}'\""
_SKIP_SCHEMES = ("mailto:", "cid:", "tel:", "data:", "#", "about:")


def normalise_domain(url: str) -> str:
    """Lower-cased host of a URL without port, userinfo or trailing dot; '' if none."""
    candidate = (url or "").strip()
    if candidate.lower().startswith("www."):
        candidate = "http://" + candidate
    try:
        host = urlsplit(candidate).hostname or ""
    except ValueError:
        return ""
    host = host.rstrip(".").lower()
    return host[4:] if host.startswith("www.") and host.count(".") > 1 else host


def is_shortened(domain: str) -> bool:
    """True when the domain belongs to a known URL shortener."""
    value = (domain or "").lower().rstrip(".")
    return value in SHORTENER_DOMAINS or any(value.endswith("." + s) for s in SHORTENER_DOMAINS)


def _clean(url: str) -> str:
    url = url.strip()
    while url and url[-1] in _TRAILING:
        url = url[:-1]
    return url


def _make(url: str, source: str) -> Optional[ExtractedURL]:
    url = _clean(url)
    lowered = url.lower()
    if not url or lowered.startswith(_SKIP_SCHEMES):
        return None
    if not lowered.startswith(("http://", "https://", "www.", "ftp://", "javascript:")):
        return None
    domain = "" if lowered.startswith("javascript:") else normalise_domain(url)
    return ExtractedURL(url=url[:2000], source=source, domain=domain, is_shortened=is_shortened(domain))


class _HrefCollector(HTMLParser):
    """Collect href / src attributes and the visible text of each anchor."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.anchors: list[dict[str, str]] = []
        self._open: list[str] = []
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        values = {k.lower(): (v or "").strip() for k, v in attrs}
        for key in ("href", "src", "action", "background"):
            if values.get(key):
                self.links.append(values[key])
        if tag.lower() == "a":
            self._open.append(values.get("href", ""))
            self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._open:
            href = self._open.pop()
            if href:
                self.anchors.append({"text": " ".join("".join(self._text).split())[:300], "href": href[:2000]})
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._open:
            self._text.append(data)


def urls_from_text(text: str, source: str = "text") -> list[ExtractedURL]:
    """URLs found by regex in a block of text."""
    found: list[ExtractedURL] = []
    for match in _URL_RE.finditer(text or ""):
        item = _make(match.group(1), source)
        if item:
            found.append(item)
    return found


def urls_from_html(html: str) -> tuple[list[ExtractedURL], list[dict[str, str]]]:
    """URLs from every href and src in the HTML, plus (anchor text, href) pairs."""
    collector = _HrefCollector()
    try:
        collector.feed(html or "")
        collector.close()
    except Exception:
        pass
    found = [item for item in (_make(link, "html_href") for link in collector.links) if item]
    return found, collector.anchors


def _open_image(image: Attachment) -> "Optional[Image.Image]":
    if not HAS_PIL or not image.raw:
        return None
    try:
        picture = Image.open(io.BytesIO(image.raw))
        picture.load()
        return picture
    except Exception:
        return None


def urls_from_qr(images: list[Attachment]) -> list[ExtractedURL]:
    """URLs encoded in QR codes inside inline images. Empty without pyzbar and Pillow."""
    if not (HAS_PYZBAR and HAS_PIL):
        return []
    found: list[ExtractedURL] = []
    for image in images:
        picture = _open_image(image)
        if picture is None:
            continue
        try:
            for symbol in zbar_decode(picture):
                data = symbol.data.decode("utf-8", "replace")
                item = _make(data, "qr") or next(iter(urls_from_text(data, "qr")), None)
                if item:
                    found.append(item)
        except Exception:
            continue
    return found


def urls_from_ocr(images: list[Attachment]) -> list[ExtractedURL]:
    """URLs in text read out of inline images. Empty without pytesseract and Pillow."""
    if not (HAS_TESSERACT and HAS_PIL):
        return []
    found: list[ExtractedURL] = []
    for image in images:
        picture = _open_image(image)
        if picture is None:
            continue
        try:
            text = pytesseract.image_to_string(picture, timeout=5)
        except Exception:
            continue
        found.extend(urls_from_text(text, "ocr"))
    return found


def extract_urls(text_body: str, html_body: str, inline_images: list[Attachment]) -> tuple[list[ExtractedURL], list[dict[str, str]]]:
    """All URLs from all four sources, de-duplicated on (url, source)."""
    html_urls, anchors = urls_from_html(html_body)
    ordered = urls_from_text(text_body) + html_urls + urls_from_qr(inline_images) + urls_from_ocr(inline_images)
    seen: set[tuple[str, str]] = set()
    unique: list[ExtractedURL] = []
    for item in ordered:
        key = (item.url, item.source)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique, anchors


def build_parsed_email(raw: bytes, source: str) -> ParsedEmail:
    """Run the header, MIME and URL parsers and return a populated ParsedEmail.

    Never raises on bad input. `raw_sha256` is computed over the bytes
    exactly as given, matching the hash taken at ingest.
    `trust_boundary_index` and `attribution` are left None for the
    forensics stage to fill in.
    """
    raw = raw or b""
    defects: list[str] = []
    try:
        msg: Message = message_from_bytes(raw, policy=policy.default)
    except Exception as exc:
        defects.append(f"message: {type(exc).__name__}")
        msg = Message()
    for defect in getattr(msg, "defects", []) or []:
        defects.append(type(defect).__name__)

    headers = parse_headers(msg)
    addresses = extract_addresses(msg)
    message_id, in_reply_to, references = parse_message_ids(headers)
    meta: dict[str, object] = {"source": source, "size_bytes": len(raw)}
    if not message_id:
        message_id = f"<missing-{sha256_of(raw)[:16]}@mailguard.invalid>"
        meta["message_id_synthesised"] = True

    try:
        text_body, html_body, attachments, inline_images = walk_mime(msg)
    except Exception as exc:
        defects.append(f"mime: {type(exc).__name__}")
        text_body, html_body, attachments, inline_images = "", "", [], []
    urls, anchors = extract_urls(text_body, html_body, inline_images)

    date_value = None
    date_raw = (headers.get("date") or [None])[0]
    if date_raw:
        try:
            date_value = parsedate_to_datetime(date_raw)
            if date_value is not None and date_value.tzinfo is None:
                date_value = date_value.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError, IndexError):
            defects.append("unparseable Date header")

    meta.update(
        {
            "date_raw": date_raw,
            "header_order": [name for name in (k.lower() for k, _ in _safe_items(msg))],
            "html_anchors": anchors,
            "parse_defects": defects,
            "url_sources_available": {
                "qr": HAS_PYZBAR and HAS_PIL,
                "ocr": HAS_TESSERACT and HAS_PIL,
            },
        }
    )

    return ParsedEmail(
        message_id=message_id,
        raw_sha256=sha256_of(raw),
        raw_bytes=raw,
        headers=headers,
        from_display=str(addresses["from_display"]),
        from_address=str(addresses["from_address"]),
        from_domain=str(addresses["from_domain"]),
        to_addresses=list(addresses["to_addresses"]),
        subject=(headers.get("subject") or [""])[0],
        text_body=text_body,
        html_body=html_body,
        urls=urls,
        attachments=attachments,
        inline_images=inline_images,
        dkim_signatures=extract_dkim_signatures(headers),
        received_chain=parse_received_chain(headers),
        references=references,
        reply_to=addresses["reply_to"],
        return_path=addresses["return_path"],
        date=date_value,
        in_reply_to=in_reply_to,
        auth_results_raw=extract_auth_results(headers),
        trust_boundary_index=None,
        attribution=None,
        meta=meta,
    )


def _safe_items(msg: Message) -> list[tuple[str, object]]:
    try:
        return list(msg.raw_items()) if hasattr(msg, "raw_items") else list(msg.items())
    except Exception:
        return []
