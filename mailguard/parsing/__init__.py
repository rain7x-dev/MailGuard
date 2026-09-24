"""Parsing package: raw bytes to a populated ParsedEmail.

`header_parser` reads addresses, ids, authentication headers and the
Received chain; `mime_parser` walks bodies and attachments; `url_extractor`
finds links in text, HTML, QR codes and images and assembles the result
with build_parsed_email(), re-exported here.

The re-export is lazy so that `python -m mailguard.parsing.header_parser`
(which runs that module's self checks) does not import it twice.
"""
from __future__ import annotations

from typing import Any

__all__ = ["build_parsed_email"]


def __getattr__(name: str) -> Any:
    if name == "build_parsed_email":
        from mailguard.parsing.url_extractor import build_parsed_email

        return build_parsed_email
    raise AttributeError(f"module 'mailguard.parsing' has no attribute {name!r}")
