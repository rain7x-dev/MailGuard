"""Ingest: the three ways raw mail enters MailGuard, and the hash taken on arrival.

HASHING HAPPENS BEFORE ANY PARSING. That ordering is the whole point of
this module. The SHA-256 of the bytes exactly as they arrived is the anchor
of the chain of custody: it lets anyone later prove that the message the
report describes is the message that was received. Hashing after
processing proves nothing about what actually arrived. It proves only that
our own output has not changed since we produced it, because parsing
decodes, normalises and re-folds, and a defence expert will ask when the
hash was taken.

So every entry door here returns the untouched bytes plus a source label,
and callers hash them with sha256_of() before handing them to the parser.

Three entry doors:

  load_eml_file   a .eml exported from a mail client or pulled from a case file
  load_from_imap  the most recent messages in a mailbox, via stdlib imaplib
  load_from_stdin raw bytes on standard input, which is how an SMTP milter
                  or a Postfix content filter hands a message over
"""
from __future__ import annotations

import hashlib
import imaplib
import logging
import os
import sys

log = logging.getLogger("mailguard.ingest")

# A message larger than this is refused. Real mail systems cap far lower,
# and an unbounded read is a trivial denial of service.
MAX_MESSAGE_BYTES: int = 50 * 1024 * 1024


def sha256_of(raw: bytes) -> str:
    """Hex SHA-256 of the raw bytes, exactly as given."""
    return hashlib.sha256(raw or b"").hexdigest()


def load_eml_file(path: str) -> tuple[bytes, str]:
    """Read a .eml file from disk, byte for byte.

    Raises FileNotFoundError or ValueError (over MAX_MESSAGE_BYTES); the
    CLI reports those to the operator rather than guessing.
    """
    size = os.path.getsize(path)
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"{path} is {size} bytes, over the {MAX_MESSAGE_BYTES} byte limit")
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw, f"file:{os.path.abspath(path)}"


def load_from_imap(
    host: str,
    user: str,
    password: str,
    mailbox: str = "INBOX",
    limit: int = 10,
) -> list[tuple[bytes, str]]:
    """Fetch the `limit` most recent messages from an IMAP mailbox over TLS.

    Messages are fetched with BODY.PEEK[] so reading them does not mark
    them as seen, and the bytes are returned exactly as the server stored
    them. Any failure (network, login, mailbox, parse) is logged as a
    warning and an empty list is returned; ingest must never crash the
    pipeline.
    """
    messages: list[tuple[bytes, str]] = []
    connection: imaplib.IMAP4_SSL | None = None
    try:
        connection = imaplib.IMAP4_SSL(host)
        connection.login(user, password)
        status, _ = connection.select(mailbox, readonly=True)
        if status != "OK":
            log.warning("IMAP: could not select mailbox %s on %s", mailbox, host)
            return []
        status, data = connection.search(None, "ALL")
        if status != "OK" or not data or not data[0]:
            return []
        ids = data[0].split()[-max(0, int(limit)):]
        for message_id in reversed(ids):
            status, parts = connection.fetch(message_id, "(BODY.PEEK[])")
            if status != "OK" or not parts:
                continue
            for part in parts:
                if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], bytes):
                    if len(part[1]) <= MAX_MESSAGE_BYTES:
                        label = f"imap://{user}@{host}/{mailbox}#{message_id.decode('ascii', 'replace')}"
                        messages.append((part[1], label))
                    break
    except Exception as exc:  # network, auth, protocol: all degrade to empty
        log.warning("IMAP ingest from %s failed: %s: %s", host, type(exc).__name__, exc)
        return messages
    finally:
        if connection is not None:
            try:
                connection.logout()
            except Exception:
                pass
    return messages


def load_from_stdin() -> tuple[bytes, str]:
    """Read one raw message from standard input, as a milter or content filter supplies it."""
    raw = sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1)
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message on stdin is over the {MAX_MESSAGE_BYTES} byte limit")
    return raw, "stdin"
