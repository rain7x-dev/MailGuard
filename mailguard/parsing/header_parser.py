"""Header parsing: addresses, message ids, authentication headers, and the Received chain.

Built on the standard library `email` package with `email.policy.default`.
Mail is attacker supplied input, so every function here tolerates malformed
headers: a value that cannot be read becomes None or an empty list, never
an exception.

THE RECEIVED CHAIN

Every server that handles a message PREPENDS a Received header, so in the
raw message they appear newest first. parse_received_chain() keeps that
order: index 0 is the TOPMOST header, the newest hop, written by the server
closest to us. Index increases going backwards in time, towards the claimed
origin.

Parsing is not trusting. Every hop is parsed identically whether our own
server wrote it or an attacker typed it. Deciding which hops can be
believed is the job of mailguard.forensics.trust_boundary.

A Received header has a loose grammar (RFC 5321 section 4.4):

    from HELO (rDNS [peer-ip]) by RECEIVER with PROTO id QUEUE-ID for <rcpt>; DATE

and every mail server writes its own dialect. The parser splits the header
into clauses at top level keywords (ignoring anything in parentheses), then
reads each clause. It handles Postfix, Sendmail, Exim, qmail, Microsoft
Exchange / Office 365 and Gmail's internal hand-offs.
"""
from __future__ import annotations

import ipaddress
import re
from datetime import timezone
from email.message import Message
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from typing import Any, Optional

from mailguard.core.models import ReceivedHop

CLAUSE_KEYWORDS: tuple[str, ...] = ("from", "by", "via", "with", "id", "for")

_WS = re.compile(r"\s+")
_BRACKET_IP = re.compile(r"\[(?:IPv6:)?([0-9A-Fa-f:.]+)\]")
_IPV4 = re.compile(r"(?<![\d.])((?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3})(?![\d.])")
_IPV6 = re.compile(r"(?<![0-9A-Za-z:])([0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7})(?![0-9A-Za-z:])")
_HELO_KV = re.compile(r"\bhelo\s*=\s*([^\s)\]]+)", re.IGNORECASE)
_HELO_WORD = re.compile(r"\bHELO\s+([^\s)]+)", re.IGNORECASE)
_HOSTNAME = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_\-]{0,62})(?:\.[A-Za-z0-9_\-]{1,63})*\.?$")
_MSGID = re.compile(r"<[^<>\s]+>")


def _unfold(value: Any) -> str:
    """Collapse header folding and runs of whitespace into single spaces."""
    return _WS.sub(" ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


def _header_str(value: Any) -> str:
    """String form of a header value, decoding RFC 2047 words when possible."""
    try:
        return _unfold(str(value))
    except Exception:
        try:
            return _unfold(getattr(value, "_parse_tree", "") or "")
        except Exception:
            return ""


def parse_headers(msg: Message) -> dict[str, list[str]]:
    """All headers, lower-cased names, values in order (headers repeat)."""
    headers: dict[str, list[str]] = {}
    try:
        items = list(msg.raw_items()) if hasattr(msg, "raw_items") else list(msg.items())
    except Exception:
        items = []
    for name, value in items:
        key = str(name).strip().lower()
        text = _header_str(value)
        if key and ("=?" in text):
            # raw_items() leaves encoded words alone; ask the policy to decode.
            try:
                decoded = msg.policy.header_fetch_parse(name, value)  # type: ignore[union-attr]
                text = _header_str(decoded)
            except Exception:
                pass
        headers.setdefault(key, []).append(text)
    return headers


def _first(headers: dict[str, list[str]], name: str) -> Optional[str]:
    values = headers.get(name)
    return values[0] if values else None


def _domain_of(address: str) -> str:
    return address.rsplit("@", 1)[1].strip().strip(">").lower().rstrip(".") if "@" in address else ""


def extract_addresses(msg: Message) -> dict[str, Any]:
    """From display name, From address and domain, To/Cc, Reply-To and Return-Path."""
    headers = parse_headers(msg)
    display, address = parseaddr(_first(headers, "from") or "")
    address = address.strip().lower()
    to_addresses: list[str] = []
    try:
        for _name, addr in getaddresses(headers.get("to", []) + headers.get("cc", [])):
            if addr and "@" in addr:
                to_addresses.append(addr.strip().lower())
    except Exception:
        pass
    reply_to_raw = _first(headers, "reply-to")
    reply_to = parseaddr(reply_to_raw)[1].strip().lower() if reply_to_raw else None
    return_path_raw = _first(headers, "return-path")
    return_path = return_path_raw.strip().strip("<>").strip() if return_path_raw else None
    return {
        "from_display": display.strip(),
        "from_address": address,
        "from_domain": _domain_of(address),
        "to_addresses": to_addresses,
        "reply_to": reply_to or None,
        "return_path": return_path or None,
    }


# ----------------------------------------------------------------------
# Received headers
# ----------------------------------------------------------------------
def clean_ip(value: Optional[str]) -> Optional[str]:
    """Normalised IP string, or None when the value is not an IP address."""
    if not value:
        return None
    candidate = value.strip().strip("[]()").strip()
    if candidate.lower().startswith("ipv6:"):
        candidate = candidate[5:]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return str(address)


def _split_clauses(text: str) -> list[tuple[str, str]]:
    """Split a Received body into (keyword, clause) pairs at parenthesis depth 0."""
    clauses: list[tuple[str, str]] = []
    depth = 0
    current: Optional[str] = None
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and (i == 0 or text[i - 1] == " "):
            for keyword in CLAUSE_KEYWORDS:
                end = i + len(keyword)
                if text[i:end].lower() == keyword and (end == len(text) or text[end] == " "):
                    if current is not None:
                        clauses.append((current, text[start:i].strip()))
                    current, start, i = keyword, end, end - 1
                    break
        i += 1
    if current is not None:
        clauses.append((current, text[start:].strip()))
    return clauses


def _first_token(clause: str) -> str:
    return clause.split("(", 1)[0].strip().split(" ", 1)[0].strip()


def _comments(clause: str) -> list[str]:
    """Top level parenthesised comments of a clause."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in clause:
        if ch == "(":
            if depth:
                buf.append(ch)
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
            if depth == 0:
                out.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        elif depth:
            buf.append(ch)
    return out


def _find_ip(text: str) -> Optional[str]:
    """First IP in text: bracketed first, then bare IPv4, then bare IPv6."""
    for pattern in (_BRACKET_IP, _IPV4):
        for match in pattern.finditer(text):
            ip = clean_ip(match.group(1))
            if ip:
                return ip
    for match in _IPV6.finditer(text):
        if match.group(1).count(":") >= 2:
            ip = clean_ip(match.group(1))
            if ip:
                return ip
    return None


def _is_hostname(value: str) -> bool:
    return bool(value) and bool(_HOSTNAME.match(value)) and clean_ip(value) is None


def parse_received_line(line: str) -> dict[str, Any]:
    """Parse one Received header value into its parts. Never raises.

    Returns helo (what the client CLAIMED to be), rdns (the reverse DNS
    name the receiving server looked up, None when it recorded "unknown"),
    ip (the socket peer the receiving server saw), by, protocol and
    timestamp. Every field the parser cannot read is None.
    """
    text = _unfold(line)
    parts: dict[str, Any] = {
        "helo": None, "rdns": None, "ip": None, "by": None,
        "protocol": None, "timestamp": None, "raw": text,
    }
    try:
        body = text
        if ";" in text:
            body, _, date_part = text.rpartition(";")
            date_text = re.sub(r"\([^)]*\)\s*$", "", date_part).strip()
            try:
                stamp = parsedate_to_datetime(date_text) if date_text else None
                if stamp is not None and stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                parts["timestamp"] = stamp
            except (TypeError, ValueError, IndexError):
                parts["timestamp"] = None
        seen: set[str] = set()
        for keyword, clause in _split_clauses(body):
            if keyword in seen:
                continue
            seen.add(keyword)
            if keyword == "from":
                token = _first_token(clause)
                comments = _comments(clause)
                joined = " ".join(comments)
                if clean_ip(token):
                    parts["ip"] = clean_ip(token)  # Exim: from [1.2.3.4] (helo=x)
                elif token:
                    parts["helo"] = token.strip("[]")
                helo = _HELO_KV.search(joined) or _HELO_WORD.search(joined)
                if helo:
                    # When the HELO is in a comment, the leading token is the rDNS.
                    if parts["helo"] and parts["helo"].lower() != "unknown":
                        parts["rdns"] = parts["helo"]
                    parts["helo"] = helo.group(1).strip("[]")
                if parts["ip"] is None:
                    parts["ip"] = _find_ip(joined)
                if parts["rdns"] is None:
                    for comment in comments:
                        first = comment.split(" ", 1)[0].strip()
                        if first.lower().startswith(("helo", "envelope")):
                            continue
                        if _is_hostname(first) and first.lower() not in ("unknown", "may", "authenticated"):
                            parts["rdns"] = first.rstrip(".").lower()
                            break
            elif keyword == "by":
                token = _first_token(clause)
                parts["by"] = (clean_ip(token) or token.strip("[]").rstrip(".").lower()) or None
            elif keyword == "with":
                parts["protocol"] = _first_token(clause) or None
    except Exception:
        pass
    return parts


def parse_received_chain(headers: dict[str, list[str]]) -> list[ReceivedHop]:
    """Every Received header as a ReceivedHop, index 0 = topmost (newest, closest to us).

    `from_host` is the name in the `from` clause as written (the HELO, or
    the reverse DNS name when the MTA puts the HELO in a comment). `trusted`
    is always False here; the trust boundary walk sets it.
    """
    hops: list[ReceivedHop] = []
    for index, line in enumerate(headers.get("received", []) or []):
        parts = parse_received_line(line)
        hops.append(
            ReceivedHop(
                index=index,
                raw=parts["raw"],
                from_host=parts["rdns"] if parts["helo"] is None else parts["helo"],
                from_ip=parts["ip"],
                by_host=parts["by"],
                timestamp=parts["timestamp"],
                trusted=False,
            )
        )
    return hops


# ----------------------------------------------------------------------
# Authentication headers and message ids
# ----------------------------------------------------------------------
def extract_auth_results(headers: dict[str, list[str]]) -> Optional[str]:
    """The topmost Authentication-Results header, written last and closest to us."""
    return _first(headers, "authentication-results")


def extract_dkim_signatures(headers: dict[str, list[str]]) -> list[str]:
    """Every DKIM-Signature header value, in order."""
    return list(headers.get("dkim-signature", []) or [])


def parse_message_ids(headers: dict[str, list[str]]) -> tuple[str, Optional[str], list[str]]:
    """(message_id, in_reply_to, references). A missing Message-ID becomes ''."""
    message_id = (_first(headers, "message-id") or "").strip()
    ids = _MSGID.findall(message_id)
    if ids:
        message_id = ids[0]
    in_reply_to_raw = _first(headers, "in-reply-to")
    in_reply_to: Optional[str] = None
    if in_reply_to_raw:
        found = _MSGID.findall(in_reply_to_raw)
        in_reply_to = found[0] if found else in_reply_to_raw.strip() or None
    references = _MSGID.findall(" ".join(headers.get("references", []) or []))
    return message_id, in_reply_to, references


if __name__ == "__main__":  # pragma: no cover - quick self checks
    from email import message_from_bytes, policy

    sample = (
        b"Received: from mx-in.example.org (mx-in.example.org [10.0.0.5]) by mx.example.org with ESMTP id A1;"
        b" Fri, 18 Sep 2026 03:41:20 +0000\r\n"
        b"Received: from smtp-out.sender.example (smtp-out.sender.example [203.0.113.19]) by mx-in.example.org"
        b" with ESMTPS id B2; Fri, 18 Sep 2026 03:41:15 +0000\r\n"
        b"From: \"Sender Name\" <Someone@Sender.Example>\r\n"
        b"To: a@example.org, b@example.org\r\n"
        b"Reply-To: other@elsewhere.example\r\n"
        b"Message-ID: <abc@sender.example>\r\n"
        b"In-Reply-To: <parent@sender.example>\r\n"
        b"References: <root@sender.example> <parent@sender.example>\r\n"
        b"Authentication-Results: mx.example.org; spf=pass; dkim=pass\r\n"
        b"DKIM-Signature: v=1; d=sender.example; s=sel1; b=xyz\r\n\r\nbody\r\n"
    )
    msg = message_from_bytes(sample, policy=policy.default)
    headers = parse_headers(msg)

    # 1. Received chain order: index 0 is the topmost header.
    chain = parse_received_chain(headers)
    assert len(chain) == 2 and chain[0].by_host == "mx.example.org" and chain[1].from_ip == "203.0.113.19"
    assert chain[0].timestamp is not None and chain[0].timestamp.tzinfo is not None

    # 2. Addresses are lower-cased and split into domain.
    addresses = extract_addresses(msg)
    assert addresses["from_address"] == "someone@sender.example" and addresses["from_domain"] == "sender.example"
    assert addresses["to_addresses"] == ["a@example.org", "b@example.org"]
    assert addresses["reply_to"] == "other@elsewhere.example"

    # 3. Message ids.
    assert parse_message_ids(headers) == (
        "<abc@sender.example>", "<parent@sender.example>", ["<root@sender.example>", "<parent@sender.example>"]
    )

    # 4. Auth headers.
    assert extract_auth_results(headers) == "mx.example.org; spf=pass; dkim=pass"
    assert extract_dkim_signatures(headers)[0].startswith("v=1; d=sender.example")

    # 5. Dialects and garbage never raise.
    exim = parse_received_line("from [198.51.100.5] (helo=client.example) by mx.host.example with esmtpsa id 1x;"
                               " Tue, 01 Sep 2026 10:00:00 +0200")
    assert exim["ip"] == "198.51.100.5" and exim["helo"] == "client.example"
    qmail = parse_received_line("from unknown (HELO evil.example) (192.0.2.44) by mail.q.example with SMTP;"
                                " 1 Sep 2026 10:00:00 -0000")
    assert qmail["helo"] == "evil.example" and qmail["ip"] == "192.0.2.44" and qmail["rdns"] is None
    gmail = parse_received_line("by 2002:a05:6a10:a0c8::2222 with SMTP id x; Tue, 1 Sep 2026 03:00:00 -0700 (PDT)")
    assert gmail["ip"] is None and gmail["by"] == "2002:a05:6a10:a0c8::2222"
    junk = parse_received_line("garbage (((( no structure")
    assert junk["by"] is None
    print("header_parser: all checks passed")
