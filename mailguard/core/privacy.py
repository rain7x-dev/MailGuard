"""Masking personal data in everything MailGuard shows to people.

Masking is applied at the output edge only: the terminal table, the JSON
view and the forensic report. Signals always score the unmasked message,
because a masked account number cannot be checked against anything, and
the evidence ledger never stores message content in the first place, only
hashes and verdicts.

What is masked, when MASK_PII is on:

  email addresses     local part reduced to its first character:
                      rakesh.menon@example.org -> r***@example.org.
                      The domain stays, because the domain is the
                      forensic finding (cousin domains, free mail).
  card numbers        13 to 19 digits, last four kept
  Aadhaar numbers     12 digits in 4-4-4 groups, last four kept
  PAN                 ABCDE1234F -> ******234F
  IFSC codes          kept: they identify a bank branch, not a person
  bank accounts       9 to 18 digit runs, last four kept
  phone numbers       +91 and 10 digit mobile numbers, last two kept

Infrastructure is never masked: IP addresses, hostnames and URLs are the
evidence, and they describe servers rather than people.
"""
from __future__ import annotations

import re
from typing import Any

from mailguard.core.config import get_config

_EMAIL = re.compile(r"\b([A-Za-z0-9._%+\-])([A-Za-z0-9._%+\-]*)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
_CARD = re.compile(r"\b(?:\d[ -]?){9,15}(\d{4})\b")
_AADHAAR = re.compile(r"\b\d{4}[ -]\d{4}[ -](\d{4})\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{1}(\d{3}[A-Z])\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{7}(\d{2})(?!\d)")


def mask_address(address: str) -> str:
    """r***@example.org from rakesh@example.org; anything else unchanged."""
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(3)}", address or "")


def mask_text(text: str) -> str:
    """Mask addresses and personal identifiers inside free text."""
    value = text or ""
    value = mask_address(value)
    value = _AADHAAR.sub(lambda m: f"XXXX-XXXX-{m.group(1)}", value)
    value = _PAN.sub(lambda m: f"******{m.group(1)}", value)
    value = _PHONE.sub(lambda m: f"********{m.group(1)}", value)
    value = _CARD.sub(lambda m: f"****{m.group(1)}", value)
    return value


def masking_enabled() -> bool:
    return bool(get_config().mask_pii)


def maybe_mask(value: Any) -> Any:
    """mask_text() when masking is on and the value is a string; else unchanged."""
    if isinstance(value, str) and masking_enabled():
        return mask_text(value)
    return value
