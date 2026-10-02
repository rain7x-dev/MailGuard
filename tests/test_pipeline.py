"""End to end and regression tests for the forensics half and the wiring.

Plain asserts, no test framework required:

    python tests/test_pipeline.py

Every test here pins down a behaviour that was once wrong, so each one
says in its docstring what used to happen.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mailguard.core.config import load_config, set_active_config
from mailguard.core.models import Attachment, Attribution, ExtractedURL
from tests.factory import make_email, make_signals, make_verdict

SAMPLES = os.path.join(ROOT, "samples")


def _offline() -> None:
    set_active_config(load_config(network_lookups=False))


def _analyse_sample(name: str, ledger=None, graph=None):
    from mailguard.cli import analyse
    from mailguard.ingest.loader import load_eml_file

    config = load_config(network_lookups=False)
    set_active_config(config)
    raw, source = load_eml_file(os.path.join(SAMPLES, name))
    return analyse(raw, source, config, ledger=ledger, graph=graph)


def _tmp(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(prefix="mailguard-test-"), name)


# ----------------------------------------------------------------------
# The three samples, end to end
# ----------------------------------------------------------------------
def test_samples_reach_the_right_action() -> None:
    """Phish BLOCKs, legitimate PASSes, and the spoofed supplier mail WARNs.

    The forged-headers mail (DMARC fail on the From domain, a fabricated
    route through our own gateway, a request to pay using new remittance
    details) used to PASS at p = 0.24.
    """
    assert _analyse_sample("cousin_domain_phish.eml")["verdict"].action == "BLOCK"
    assert _analyse_sample("legitimate.eml")["verdict"].action == "PASS"
    forged = _analyse_sample("forged_headers.eml")["verdict"]
    assert forged.action in ("WARN", "BLOCK"), f"spoofed mail passed at p={forged.probability}"
    assert "x1*x6" in forged.contributions, "the spoofing interaction must be itemised"


def test_ledger_and_graph_are_written_by_the_pipeline() -> None:
    """The CLI used to produce a verdict without ever touching the ledger or the graph."""
    from mailguard.evidence.ledger import EvidenceLedger
    from mailguard.graph.campaign_graph import CampaignGraph

    ledger = EvidenceLedger(_tmp("ledger.jsonl"))
    graph = CampaignGraph()
    result = _analyse_sample("cousin_domain_phish.eml", ledger=ledger, graph=graph)
    entries = ledger.entries_for_case(result["case_id"])
    assert [e["type"] for e in entries] == ["raw_seal", "parsed", "signals", "verdict"]
    assert entries[0]["payload"]["raw_sha256"] == result["raw_sha256"], "seal the bytes as received"
    assert result["verdict"].evidence_hash == ledger.head()
    assert result["verdict"].campaign_id, "a BLOCK verdict must be filed into a campaign"

    # Analysing the same bytes again must not crash on the seal-first rule,
    # and must leave the chain intact.
    again = _analyse_sample("cousin_domain_phish.eml", ledger=ledger, graph=graph)
    assert again["ledger_head"] and ledger.verify_chain() == (True, None)
    assert ledger.entries_for_case(result["case_id"])[4]["type"] == "reanalysis"

    # A PASS verdict is sealed but never filed as a campaign.
    legit = _analyse_sample("legitimate.eml", ledger=ledger, graph=graph)
    assert legit["verdict"].campaign_id is None


# ----------------------------------------------------------------------
# Trust boundary semantics, shared by forensics and the report
# ----------------------------------------------------------------------
def test_report_draws_the_boundary_above_the_first_untrusted_hop() -> None:
    """The report used to read the boundary as the LAST trusted hop, one off from forensics."""
    from mailguard.evidence.pdf_report import _boundary_note, _build_html, _report_model

    assert _boundary_note(1, 4).startswith("Hop 0 was recorded")
    assert "Hops 0 to 2 were" in _boundary_note(3, 4)
    assert "inside our own network" in _boundary_note(None, 2)

    result = _analyse_sample("forged_headers.eml")
    model = _report_model(result["email"], result["verdict"])
    page = open(_build_html(model, _tmp("r.html")), encoding="utf-8").read()
    rule, hop0, hop1 = page.index("TRUST BOUNDARY"), page.index("<b>hop 0</b>"), page.index("<b>hop 1</b>")
    assert hop0 < rule < hop1, "hop 1 is the forged one and must sit below the rule"


def test_factory_agrees_with_the_forensics_walk() -> None:
    email = make_email()
    first_untrusted = next(h.index for h in email.received_chain if not h.trusted)
    assert email.trust_boundary_index == first_untrusted


def test_report_custody_is_scoped_to_the_case() -> None:
    """With several messages in one ledger, the report used to show the first message's seal."""
    from mailguard.evidence.ledger import EvidenceLedger, case_id_for

    ledger = EvidenceLedger(_tmp("ledger.jsonl"))
    ledger.seal_raw("a" * 64, "test", "tester")
    ledger.seal_raw("b" * 64, "test", "tester")
    report = ledger.verification_report(case_id_for("b" * 64))
    seal_b = ledger.entries_for_case(case_id_for("b" * 64))[0]
    assert report["seal_hash"] == seal_b["entry_hash"]
    assert report["case_entry_count"] == 1 and report["entry_count"] == 2


# ----------------------------------------------------------------------
# x1: forged Authentication-Results
# ----------------------------------------------------------------------
def test_authentication_results_below_the_boundary_are_not_ours() -> None:
    """An attacker-typed 'Authentication-Results: mx.example.org; ... pass' used to be trusted."""
    from mailguard.forensics.attribution import attribute
    from mailguard.forensics.trust_boundary import annotate_chain
    from mailguard.parsing.url_extractor import build_parsed_email
    from mailguard.signals import x1_authentication

    raw = (
        b"Received: from unknown (unknown [198.51.100.9]) by mx.example.org with ESMTP id A;"
        b" Mon, 21 Sep 2026 10:00:00 +0000\r\n"
        b"Received: from x (x [192.0.2.1]) by relay.attacker.example with ESMTP id B;"
        b" Mon, 21 Sep 2026 09:59:00 +0000\r\n"
        b"Authentication-Results: mx.example.org; spf=pass smtp.mailfrom=ceo@bank.example;"
        b" dkim=pass header.d=bank.example; dmarc=pass\r\n"
        b"From: CEO <ceo@bank.example>\r\nTo: a@example.org\r\nSubject: hi\r\n\r\nbody\r\n"
    )
    _offline()
    email = build_parsed_email(raw, "test")
    annotate_chain(email, ["mx.example.org"])
    email.attribution = attribute(email)
    result = x1_authentication.run(email)
    assert result.details.get("trusted_source") is False, "a header the sender wrote is not ours"


# ----------------------------------------------------------------------
# x7: attachments and the threat feed
# ----------------------------------------------------------------------
def _attachment(name: str, content_type: str, raw: bytes) -> Attachment:
    from mailguard.parsing.mime_parser import detect_magic_type

    return Attachment(name, content_type, len(raw), hashlib.sha256(raw).hexdigest(), detect_magic_type(raw), raw)


def test_clean_office_files_are_not_flagged() -> None:
    """A plain .docx used to be a MIME mismatch, and every legacy .doc a macro document."""
    from mailguard.signals.x7_attachment_url import attachment_risk

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<w/>")
    docx = _attachment("report.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       buffer.getvalue())
    risk, findings = attachment_risk(docx)
    assert risk == 0.0 and not findings["mime_mismatch"] and not findings["archive"], findings

    ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600
    risk, findings = attachment_risk(_attachment("minutes.doc", "application/msword", ole))
    assert not findings["macro"], "an OLE2 container alone is not a macro"

    with_vba = ole + "_VBA_PROJECT".encode("utf-16-le") + b"\x00" * 64
    risk, findings = attachment_risk(_attachment("minutes.doc", "application/msword", with_vba))
    assert findings["macro"] and risk >= 0.55


def test_shortcut_files_are_executable() -> None:
    from mailguard.signals.x7_attachment_url import attachment_risk

    lnk = b"\x4c\x00\x00\x00\x01\x14\x02\x00" + b"\x00" * 64
    _risk, findings = attachment_risk(_attachment("photo.jpg", "image/jpeg", lnk))
    assert findings["executable"]


def test_phishtank_feed_hit_is_scored() -> None:
    from mailguard.signals import x7_attachment_url as x7

    path = _tmp("online-valid.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("phish_id,url,phish_detail_url\n1,http://login-verify.example/hdfc/,x\n")
    old = x7.PHISHTANK_PATH
    x7.PHISHTANK_PATH = path
    try:
        assert x7.phish_feed_match("http://login-verify.example/hdfc") == 1.0
        assert x7.phish_feed_match("http://login-verify.example/other") == 0.6
        assert x7.phish_feed_match("http://unrelated.example/") == 0.0
        email = make_email(urls=[ExtractedURL("http://login-verify.example/hdfc/", "text", "login-verify.example")],
                           attachments=[], html_body="")
        result = x7.run(email)
        assert result.score >= 0.9 and "PhishTank" in result.evidence_row
    finally:
        x7.PHISHTANK_PATH = old


# ----------------------------------------------------------------------
# x2, x3, x4
# ----------------------------------------------------------------------
def test_executive_impersonation() -> None:
    from mailguard.signals import x2_identity as x2

    old = (x2.EXECUTIVE_NAMES, x2.ORG_DOMAINS)
    x2.EXECUTIVE_NAMES, x2.ORG_DOMAINS = ["Rajesh Kumar"], ["victim.example"]
    try:
        assert x2.executive_impersonation("Rajesh Kumar (CEO)", "gmail.com")[0] == 1.0
        assert x2.executive_impersonation("Rajesh Kumar", "victim.example")[0] == 0.0
        assert x2.executive_impersonation("Rajesh", "gmail.com")[0] == 0.0
        email = make_email(from_display="Rajesh Kumar", from_address="ceo.office@gmail.com",
                           from_domain="gmail.com", reply_to=None)
        result = x2.run(email)
        assert result.details["executive_impersonation"] == 1.0
        assert "executive" in result.evidence_row
    finally:
        x2.EXECUTIVE_NAMES, x2.ORG_DOMAINS = old


def test_hosting_asn_scores_in_x3_not_x6() -> None:
    """The PPT puts hosting ASN under Infrastructure; it used to score in x6 only."""
    from mailguard.signals import x3_infrastructure, x6_origin_geo

    _offline()
    email = make_email(attribution=Attribution(tier=2, tier_name="provider_bounded",
                                               boundary_ip="203.0.113.19", asn="AS64500",
                                               isp="Example Hosting VPS", notes="hosting network (x)"))
    email.meta["attribution_lookup"] = {"asn": "AS64500", "isp": "Example Hosting VPS", "is_datacentre": True}
    x3 = x3_infrastructure.run(email)
    assert x3.status == "ok" and x3.details["features"]["hosting_asn"] == 1.0, x3.evidence_row
    x6 = x6_origin_geo.run(email)
    assert not any("datacentre" in reason for reason in x6.details["risk_reasons"])


def test_dormant_account_is_measured() -> None:
    """A naive last_seen in the history store used to zero this feature silently."""
    from mailguard.signals import x4_sender_baseline as x4

    path = _tmp("history.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"senders": {"accounts@hdfc-verify.example": {
            "count": 10, "last_seen": "2025-01-01T09:00:00", "hours": [9] * 10, "bodies": ["hello"] * 10}}}, fh)
    old = x4.HISTORY_PATH
    x4.HISTORY_PATH = path
    try:
        assert x4.run(make_email()).details["days_since_last_contact"] == 1.0
    finally:
        x4.HISTORY_PATH = old


# ----------------------------------------------------------------------
# Attribution, trace map, graph, privacy
# ----------------------------------------------------------------------
def test_tor_exit_list_makes_tier_three() -> None:
    from mailguard.forensics.attribution import attribute
    from mailguard.forensics.trust_boundary import annotate_chain
    from mailguard.parsing.url_extractor import build_parsed_email

    exits = _tmp("exits.txt")
    with open(exits, "w", encoding="utf-8") as fh:
        fh.write("# tor exits\n198.51.100.66\n")
    config = load_config(network_lookups=False, tor_exit_list_path=exits)
    raw = (b"Received: from x (unknown [198.51.100.66]) by mx.example.org with ESMTP id A;"
           b" Mon, 21 Sep 2026 10:00:00 +0000\r\nFrom: a@b.example\r\n\r\nbody\r\n")
    email = build_parsed_email(raw, "test")
    annotate_chain(email, config.trusted_hosts)
    attribution = attribute(email, config)
    assert attribution.tier == 3 and attribution.is_vpn_or_tor


def test_trace_map_never_pins_above_tier_one() -> None:
    from mailguard.evidence.trace_map import build_trace_map, pin_for

    provider = make_email()
    assert pin_for(provider) is None
    page = open(build_trace_map(provider, make_verdict(provider), _tmp("m.html")), encoding="utf-8").read()
    assert "leaflet" not in page.lower() and "No location is plotted" in page

    direct = make_email(attribution=Attribution(tier=1, tier_name="direct_ip", boundary_ip="198.51.100.5",
                                                country="IN", city="Pune", latitude=18.52, longitude=73.86))
    assert pin_for(direct) is not None
    page = open(build_trace_map(direct, None, _tmp("m.html")), encoding="utf-8").read()
    assert "L.map" in page and "Pune" in page


def test_default_dkim_selectors_do_not_link_campaigns() -> None:
    from mailguard.graph.campaign_graph import CampaignGraph

    graph = CampaignGraph()
    signals = make_signals()
    signals[0].details["dkim_selector"] = "selector1"
    one = make_email(message_id="<1@a.example>", from_domain="a.example", reply_to=None, urls=[],
                     attribution=None, raw_sha256="1" * 64)
    two = make_email(message_id="<2@b.example>", from_domain="b.example", reply_to=None, urls=[],
                     attribution=None, raw_sha256="2" * 64)
    first = graph.add_verdict(one, make_verdict(one, signals))
    second = graph.add_verdict(two, make_verdict(two, signals))
    assert first != second, "Microsoft 365's default selector is not an operator fingerprint"


def test_retention_prunes_old_cases() -> None:
    from mailguard.graph.campaign_graph import CampaignGraph

    graph = CampaignGraph()
    old = make_email(date=datetime.now(timezone.utc) - timedelta(days=400))
    graph.add_verdict(old, make_verdict(old))
    assert graph.prune(365) == 1 and graph.stats()["campaigns"] == 0


def test_masking() -> None:
    from mailguard.core.privacy import mask_text

    text = "Pay 1234 5678 9012 3456 from rakesh.menon@example.org, PAN ABCDE1234F, call 9876543210"
    masked = mask_text(text)
    for secret in ("rakesh.menon", "1234 5678 9012", "ABCDE", "98765432"):
        assert secret not in masked, f"{secret} leaked: {masked}"
    assert "@example.org" in masked and "3456" in masked, "the domain and last digits stay"


def test_contract_names_match_the_ppt() -> None:
    from mailguard.fusion.ebm_fusion import SIGNAL_LABELS
    from mailguard.signals import discover_signals

    names = {m.SIGNAL_ID: m.SIGNAL_NAME for m in discover_signals()}
    for sid, label in SIGNAL_LABELS.items():
        assert names.get(sid) == label, f"{sid}: fusion says {label!r}, signal says {names.get(sid)!r}"


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def _all_tests() -> list[tuple[str, Callable[[], None]]]:
    module = sys.modules[__name__]
    return [(n, getattr(module, n)) for n in sorted(dir(module)) if n.startswith("test_") and callable(getattr(module, n))]


def main() -> int:
    passed, failed = 0, 0
    print("MailGuard pipeline and regression tests")
    print("=" * 78)
    for name, test in _all_tests():
        try:
            test()
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            print("      " + "\n      ".join(traceback.format_exc().strip().splitlines()[-6:]))
        else:
            passed += 1
            print(f"ok    {name}")
    print("-" * 78)
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
