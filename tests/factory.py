"""Fully populated fake objects, so this half of MailGuard is testable alone.

The ingest and parsing half of the project is being written independently,
so nothing here may depend on it. These builders produce a ParsedEmail, a
set of SignalResults and a Verdict that are realistic enough to drive the
fusion layer, the campaign graph, the ledger and the PDF report end to
end without a single real mail.

The mail they describe is one specific attack, chosen because it is the
case the project exists for: a cousin domain impersonating a bank, a
display name that claims the brand, a Reply-To pointing at free mail, a
macro bearing spreadsheet, a punycode landing page, and an
authentication result that PASSES. It is caught anyway.

RECEIVED CHAIN ORDER, since the report draws a line through it: index 0 is
the topmost Received header, which is the most recent hop, our own MX.
Index increases going backwards in time towards the origin.
trust_boundary_index is the last index we control; anything with a higher
index was written by a host we do not trust, and can be a lie.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any

# tests/ is not a package, so make the repo root importable when this file is
# imported directly (python tests/test_fusion.py, or from a signal demo).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mailguard.core.models import (
    Attachment,
    Attribution,
    ExtractedURL,
    ParsedEmail,
    ReceivedHop,
    SignalResult,
    Verdict,
)

# A fixed date, so reports and tests are reproducible.
SENT_AT = datetime(2026, 9, 18, 3, 41, 12, tzinfo=timezone.utc)


def macro_spreadsheet_bytes() -> bytes:
    """A real OOXML zip carrying xl/vbaProject.bin, built in memory.

    Built rather than hard coded so x7's macro detection is exercised
    against genuine zip structure: the point of that check is that it
    looks inside the container instead of trusting the extension.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"/>',
        )
        archive.writestr("xl/workbook.xml", '<?xml version="1.0"?><workbook/>')
        archive.writestr("xl/worksheets/sheet1.xml", '<?xml version="1.0"?><worksheet/>')
        # The payload. Content is irrelevant; presence is the finding.
        archive.writestr("xl/vbaProject.bin", b"\x00\x01\x02VBA-STUB-FOR-TESTS")
    return buffer.getvalue()


def make_email(**overrides: Any) -> ParsedEmail:
    """A realistic cousin domain phishing mail.

    Four Received hops, trust boundary at index 2, tier 2 attribution, two
    URLs and one macro bearing attachment. Any field can be overridden by
    keyword, which is how the tests build variants.
    """
    attachment_raw = macro_spreadsheet_bytes()
    attachment = Attachment(
        filename="Invoice_HDFC_44718.xlsm",
        content_type="application/vnd.ms-excel.sheet.macroenabled.12",
        size_bytes=len(attachment_raw),
        sha256=hashlib.sha256(attachment_raw).hexdigest(),
        magic_type="zip",
        raw=attachment_raw,
    )

    text_body = (
        "Dear Accounts Team,\r\n\r\n"
        "Please note that our bank details have changed with immediate effect. "
        "All future payments should be made to the account below. Kindly update "
        "your records and process the pending payment of INR 12,45,000 today "
        "itself, as our old account is frozen pending an audit.\r\n\r\n"
        "The revised invoice is attached. You can also confirm the new details "
        "on our secure portal, please click here to verify your account.\r\n\r\n"
        "I am travelling and cannot take calls, so kindly keep this "
        "confidential and reply only to this address.\r\n\r\n"
        "Regards,\r\n"
        "Rakesh Menon\r\n"
        "Accounts Receivable, HDFC Bank\r\n"
    )
    html_body = (
        "<html><body><p>Dear Accounts Team,</p>"
        "<p>Please note that our <b>bank details have changed</b> with immediate "
        "effect. All future payments should be made to the account below.</p>"
        '<p>Confirm the new details here: <a href="http://xn--hdfc-4n1e.in/'
        'secure/login/verify?ref=44718">www.hdfc.in/secure</a></p>'
        '<p><a href="https://bit.ly/3xZqR7v">View the revised invoice</a></p>'
        "<p>Regards,<br>Rakesh Menon<br>Accounts Receivable, HDFC Bank</p>"
        "</body></html>"
    )

    urls = [
        ExtractedURL(
            url="http://xn--hdfc-4n1e.in/secure/login/verify?ref=44718",
            source="html_href",
            domain="xn--hdfc-4n1e.in",
            is_shortened=False,
        ),
        ExtractedURL(
            url="https://bit.ly/3xZqR7v",
            source="html_href",
            domain="bit.ly",
            is_shortened=True,
            final_url=None,
        ),
    ]

    # Index 0 is the most recent hop (our MX). Hops 0 to 2 are inside
    # infrastructure we control and are therefore trusted; hop 3 was
    # written by a host we do not control and can say anything it likes.
    received_chain = [
        ReceivedHop(
            index=0,
            raw=(
                "from mx-in-02.victim.example (mx-in-02.victim.example [198.51.100.12]) "
                "by mail.victim.example with ESMTPS id 4c1f9a; "
                "Fri, 18 Sep 2026 03:41:20 +0000"
            ),
            from_host="mx-in-02.victim.example",
            from_ip="198.51.100.12",
            by_host="mail.victim.example",
            timestamp=SENT_AT + timedelta(seconds=8),
            trusted=True,
        ),
        ReceivedHop(
            index=1,
            raw=(
                "from filter-07.victim.example (filter-07.victim.example [198.51.100.7]) "
                "by mx-in-02.victim.example with ESMTP id 9b23ee; "
                "Fri, 18 Sep 2026 03:41:18 +0000"
            ),
            from_host="filter-07.victim.example",
            from_ip="198.51.100.7",
            by_host="mx-in-02.victim.example",
            timestamp=SENT_AT + timedelta(seconds=6),
            trusted=True,
        ),
        ReceivedHop(
            index=2,
            raw=(
                "from smtp-out-19.bulkrelay.example (smtp-out-19.bulkrelay.example "
                "[203.0.113.19]) by filter-07.victim.example with ESMTPS id 71ad02; "
                "Fri, 18 Sep 2026 03:41:15 +0000"
            ),
            from_host="smtp-out-19.bulkrelay.example",
            from_ip="203.0.113.19",
            by_host="filter-07.victim.example",
            timestamp=SENT_AT + timedelta(seconds=3),
            trusted=True,
        ),
        ReceivedHop(
            index=3,
            raw=(
                "from localhost (unknown [10.8.0.14]) by smtp-out-19.bulkrelay.example "
                "with ESMTPA id 0f44b1; Fri, 18 Sep 2026 03:41:12 +0000"
            ),
            from_host="localhost",
            from_ip="10.8.0.14",
            by_host="smtp-out-19.bulkrelay.example",
            timestamp=SENT_AT,
            trusted=False,
        ),
    ]

    raw_bytes = (
        b"From: \"HDFC Bank Accounts\" <accounts@hdfc-verify.example>\r\n"
        b"To: finance@victim.example\r\n"
        b"Subject: URGENT: revised payment instructions, account change\r\n"
        b"Reply-To: hdfc.recovery.desk01@gmail.com\r\n"
        b"\r\n" + text_body.encode("utf-8")
    )

    defaults: dict[str, Any] = {
        "message_id": "<20260918034112.44718.9c2f@hdfc-verify.example>",
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_bytes": raw_bytes,
        "headers": {
            "from": ['"HDFC Bank Accounts" <accounts@hdfc-verify.example>'],
            "to": ["finance@victim.example"],
            "subject": ["URGENT: revised payment instructions, account change"],
            "reply-to": ["hdfc.recovery.desk01@gmail.com"],
            "return-path": ["<bounce-44718@bulkrelay.example>"],
            "date": ["Fri, 18 Sep 2026 03:41:12 +0000"],
            "message-id": ["<20260918034112.44718.9c2f@hdfc-verify.example>"],
            "dkim-signature": [
                "v=1; a=rsa-sha256; d=hdfc-verify.example; s=mailer01; "
                "h=from:to:subject:date; bh=6Qk1...; b=Vd9x..."
            ],
            "authentication-results": [
                "mx-in-02.victim.example; spf=pass "
                "smtp.mailfrom=bounce-44718@bulkrelay.example; "
                "dkim=pass header.d=hdfc-verify.example; dmarc=pass"
            ],
        },
        "from_display": "HDFC Bank Accounts",
        "from_address": "accounts@hdfc-verify.example",
        "from_domain": "hdfc-verify.example",
        "to_addresses": ["finance@victim.example"],
        "subject": "URGENT: revised payment instructions, account change",
        "text_body": text_body,
        "html_body": html_body,
        "urls": urls,
        "attachments": [attachment],
        "inline_images": [],
        "dkim_signatures": [
            "v=1; a=rsa-sha256; d=hdfc-verify.example; s=mailer01; "
            "h=from:to:subject:date; bh=6Qk1...; b=Vd9x..."
        ],
        "received_chain": received_chain,
        "references": [],
        "reply_to": "hdfc.recovery.desk01@gmail.com",
        "return_path": "<bounce-44718@bulkrelay.example>",
        "date": SENT_AT,
        "in_reply_to": None,
        "auth_results_raw": (
            "mx-in-02.victim.example; spf=pass "
            "smtp.mailfrom=bounce-44718@bulkrelay.example; "
            "dkim=pass header.d=hdfc-verify.example; dmarc=pass"
        ),
        "trust_boundary_index": 2,
        "attribution": Attribution(
            tier=2,
            tier_name="Infrastructure attributed",
            boundary_ip="203.0.113.19",
            asn="AS200000",
            country="NL",
            isp="Bulk Relay Hosting BV",
            is_vpn_or_tor=False,
            notes=(
                "Boundary is a shared outbound relay. The relay's own inbound hop "
                "claims 10.8.0.14, an RFC1918 address written by a host we do not "
                "control, so the true origin is not established. Attribution stops "
                "at the relay operator."
            ),
        ),
        "meta": {"ingest_source": "smtp://mx-in-02.victim.example", "operator": "analyst.rk"},
    }
    defaults.update(overrides)
    return ParsedEmail(**defaults)


def make_signals() -> list[SignalResult]:
    """Seven signal results for the mail make_email() describes.

    x1 PASSES, which is the whole point of the example: the attacker owns
    hdfc-verify.example and publishes correct SPF and DKIM records for it.
    x4 ABSTAINS, because a four day old domain has no sender baseline, so
    the missing data path through fusion is exercised by default.
    """
    return [
        SignalResult(
            signal_id="x1",
            name="Authentication",
            score=0.05,
            status="ok",
            evidence_row="SPF pass, DKIM pass (d=hdfc-verify.example, s=mailer01), DMARC pass",
            details={
                "spf": "pass",
                "dkim": "pass",
                "dmarc": "pass",
                "dkim_domain": "hdfc-verify.example",
                "dkim_selector": "mailer01",
                "note": "authenticated correctly for a domain the attacker owns",
            },
        ),
        SignalResult(
            signal_id="x2",
            name="Identity",
            score=0.91,
            status="ok",
            evidence_row=(
                "cousin domain of hdfc.in, brand label 'hdfc' embedded in an unrelated "
                "domain, display name claims 'hdfc', reply-to free mail at gmail.com, "
                "first contact"
            ),
            details={
                "cousin_domain_score": 0.90,
                "matched_domain": "hdfc.in",
                "mechanism": "brand label 'hdfc' embedded in an unrelated domain",
                "display_name_brand_mismatch": 1.0,
                "reply_to_freemail": 1.0,
                "first_contact": 1.0,
                "from_domain": "hdfc-verify.example",
            },
        ),
        SignalResult(
            signal_id="x3",
            name="Header and Route",
            score=0.44,
            status="ok",
            evidence_row=(
                "Return-Path domain bulkrelay.example does not match From domain, "
                "last untrusted hop claims an RFC1918 address"
            ),
            details={"return_path_mismatch": True, "private_ip_claim": "10.8.0.14"},
        ),
        SignalResult(
            signal_id="x4",
            name="Sender Baseline",
            score=0.0,
            status="abstain",
            evidence_row="no baseline: 0 prior messages, 5 required",
            details={
                "history_count": 0,
                "min_required": 5,
                "abstain_reason": "insufficient sender history",
            },
        ),
        SignalResult(
            signal_id="x5",
            name="Intent",
            score=0.86,
            status="ok",
            evidence_row=(
                "urgency (subject): urgent; bank detail change: all future payments "
                "should be made to; payment redirection: our old account is frozen; "
                "secrecy: keep this confidential"
            ),
            details={
                "bank_detail_change": 1.0,
                "payment_redirection": 0.67,
                "urgency_subject": 0.33,
                "secrecy_pressure": 0.33,
                "authority_pressure": 0.33,
                "scorer": "keyword-heuristic",
            },
        ),
        SignalResult(
            signal_id="x6",
            name="Origin and Infrastructure",
            score=0.58,
            status="ok",
            evidence_row=(
                "boundary 203.0.113.19 on AS200000 (Bulk Relay Hosting BV, NL), "
                "shared outbound relay, domain registered 4 days ago"
            ),
            details={
                "asn": "AS200000",
                "country": "NL",
                "isp": "Bulk Relay Hosting BV",
                "domain_age_days": 4,
            },
        ),
        SignalResult(
            signal_id="x7",
            name="Attachment and URL",
            score=0.78,
            status="ok",
            evidence_row=(
                "Invoice_HDFC_44718.xlsm: macro document (vbaProject.bin present inside "
                "the OOXML zip); link text says www.hdfc.in but points to "
                "xn--hdfc-4n1e.in; shortened link, not resolved"
            ),
            details={
                "macro_office_document": 1.0,
                "punycode_url": 1.0,
                "anchor_text_mismatch": 1.0,
                "shortened_url": 1.0,
                "url_domains": ["bit.ly", "xn--hdfc-4n1e.in"],
            },
        ),
    ]


def make_verdict(email: ParsedEmail | None = None, signals: list[SignalResult] | None = None) -> Verdict:
    """A complete verdict with real contributions from the fusion layer.

    The numbers come from fuse() rather than being written out by hand, so
    a test that reads this verdict is testing the model and not a fixture
    that has drifted away from it.
    """
    from mailguard.fusion.ebm_fusion import fuse

    email = email if email is not None else make_email()
    signals = signals if signals is not None else make_signals()
    verdict = fuse(signals, email)
    verdict.campaign_id = "CMP-8E68BE8084"
    verdict.evidence_hash = hashlib.sha256(
        f"{email.raw_sha256}|{verdict.probability}|{verdict.action}".encode("utf-8")
    ).hexdigest()
    return verdict


if __name__ == "__main__":  # pragma: no cover - quick look at the fixtures
    fake_email = make_email()
    fake_verdict = make_verdict(fake_email)
    print(f"message    : {fake_email.subject}")
    print(f"from       : {fake_email.from_display} <{fake_email.from_address}>")
    print(f"attachments: {[a.filename for a in fake_email.attachments]}")
    print(f"urls       : {[u.domain for u in fake_email.urls]}")
    print(f"hops       : {len(fake_email.received_chain)}, boundary at index "
          f"{fake_email.trust_boundary_index}")
    print(f"verdict    : {fake_verdict.action} {fake_verdict.verdict_class} "
          f"p={fake_verdict.probability:.4f}")
    print("contributions (log-odds points):")
    for key, value in sorted(fake_verdict.contributions.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<12} {value:+.3f}")
