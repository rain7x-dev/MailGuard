"""Shared data contract for MailGuard AI.

Every module in the project passes these objects around. Do not change
field names without coordinating, other modules depend on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Attachment:
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    magic_type: str
    raw: bytes = b""


@dataclass
class ExtractedURL:
    url: str
    source: str
    domain: str
    is_shortened: bool = False
    final_url: Optional[str] = None


@dataclass
class ReceivedHop:
    index: int
    raw: str
    from_host: Optional[str] = None
    from_ip: Optional[str] = None
    by_host: Optional[str] = None
    timestamp: Optional[datetime] = None
    trusted: bool = False


@dataclass
class Attribution:
    tier: int
    tier_name: str
    boundary_ip: Optional[str] = None
    asn: Optional[str] = None
    country: Optional[str] = None
    isp: Optional[str] = None
    is_vpn_or_tor: bool = False
    notes: str = ""


@dataclass
class ParsedEmail:
    message_id: str
    raw_sha256: str
    raw_bytes: bytes
    headers: dict[str, list[str]]
    from_display: str
    from_address: str
    from_domain: str
    to_addresses: list[str]
    subject: str
    text_body: str
    html_body: str
    urls: list[ExtractedURL]
    attachments: list[Attachment]
    inline_images: list[Attachment]
    dkim_signatures: list[str]
    received_chain: list[ReceivedHop]
    references: list[str]
    reply_to: Optional[str] = None
    return_path: Optional[str] = None
    date: Optional[datetime] = None
    in_reply_to: Optional[str] = None
    auth_results_raw: Optional[str] = None
    trust_boundary_index: Optional[int] = None
    attribution: Optional[Attribution] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalResult:
    signal_id: str
    name: str
    score: float
    status: str
    evidence_row: str
    details: dict[str, Any] = field(default_factory=dict)
    model_backed: bool = False


@dataclass
class Verdict:
    probability: float
    verdict_class: str
    action: str
    contributions: dict[str, float]
    signals: list[SignalResult]
    attribution: Optional[Attribution] = None
    campaign_id: Optional[str] = None
    evidence_hash: Optional[str] = None
