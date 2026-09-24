"""Ingest package: read raw message bytes from a file, an IMAP mailbox or stdin, and hash them first."""
from __future__ import annotations

from mailguard.ingest.loader import load_eml_file, load_from_imap, load_from_stdin, sha256_of

__all__ = ["load_eml_file", "load_from_imap", "load_from_stdin", "sha256_of"]
