"""The forensic report: the document an investigator actually receives.

Everything else in MailGuard produces numbers. This module produces the
artefact a human reads, signs and attaches to a case, so it is written to
be defensible rather than decorative. Three rules shape it:

1. Every point of the verdict is traceable. The evidence table prints one
   row per signal with its contribution in log-odds points, ordered
   strongest first, and the model arithmetic block underneath shows the
   interaction terms and the intercept so the numbers visibly add up to
   the probability at the top of the page. That is what the explainable
   model was chosen for; hiding it here would waste the choice.

2. An abstaining signal is never printed as a zero. It is printed as
   "abstain, not scored", because a reader who mistakes missing data for
   a clean result draws exactly the wrong conclusion.

3. The report never claims more attribution than the evidence supports.
   For tier 2 and tier 3 origins it does not print a confident location,
   and for tier 3 it says plainly that the origin was concealed and that
   the concealment is itself the finding.

reportlab renders the PDF. If reportlab is not installed the same report
is written as a self-contained HTML file next to the requested path, and
the function returns that path, so generating a report never simply
fails.
"""
from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from typing import Any, Optional

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import ParsedEmail, SignalResult, Verdict
from mailguard.evidence.ledger import GENESIS_HASH, EvidenceLedger, case_id_for

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

TITLE = "MailGuard AI Forensic Report"

# Action colours, used by both renderers.
ACTION_COLOURS: dict[str, str] = {
    "BLOCK": "#B00020",   # red
    "WARN": "#B26A00",    # amber
    "PASS": "#1B7A3D",    # green
}

# Statuses that mean the signal produced no usable score.
UNUSABLE_STATUSES: frozenset[str] = frozenset({"abstain", "error", "skipped", "unavailable"})


# ----------------------------------------------------------------------
# Content assembly, shared by the PDF and the HTML renderer so the two
# can never drift apart.
# ----------------------------------------------------------------------
def _is_abstain(signal: SignalResult) -> bool:
    return (signal.status or "").lower() in UNUSABLE_STATUSES


def _evidence_rows(verdict: Verdict) -> list[dict[str, Any]]:
    """One row per signal, strongest contribution first."""
    rows: list[dict[str, Any]] = []
    for signal in verdict.signals or []:
        points = float(verdict.contributions.get(signal.signal_id, 0.0))
        abstained = _is_abstain(signal)
        rows.append(
            {
                "signal_id": signal.signal_id,
                "name": signal.name,
                "score": "abstain, not scored" if abstained else f"{float(signal.score):.2f}",
                "status": (signal.status or "").lower() or "unknown",
                "points": points,
                "points_text": f"{points:+.2f}",
                "evidence": signal.evidence_row or "",
                "abstained": abstained,
                "model_backed": bool(signal.model_backed),
            }
        )
    rows.sort(key=lambda row: row["points"], reverse=True)
    return rows


def _arithmetic_rows(verdict: Verdict) -> list[tuple[str, str]]:
    """Interaction terms, intercept and the total, as printable pairs."""
    rows: list[tuple[str, str]] = []
    for key, value in sorted(verdict.contributions.items(), key=lambda kv: -kv[1]):
        if "*" not in key:
            continue
        first, second = key.split("*", 1)
        rows.append((f"interaction {first} with {second}", f"{value:+.2f}"))
    if "_intercept" in verdict.contributions:
        rows.append(
            ("base rate (intercept)", f"{float(verdict.contributions['_intercept']):+.2f}")
        )
    total = sum(float(v) for v in verdict.contributions.values())
    rows.append(("total, log-odds of fraud", f"{total:+.2f}"))
    rows.append(("probability after calibration", f"{verdict.probability:.4f}"))
    return rows


def _attribution_rows(email: ParsedEmail, verdict: Verdict) -> tuple[list[tuple[str, str]], list[str]]:
    """Attribution facts, plus the caveats the tier forces us to print."""
    attribution = verdict.attribution or email.attribution
    if attribution is None:
        return (
            [("Origin attribution", "not performed, no attribution attached to this message")],
            [],
        )

    rows: list[tuple[str, str]] = [
        ("Tier", f"{attribution.tier} - {attribution.tier_name}"),
        ("Boundary IP", attribution.boundary_ip or "not established"),
        ("ASN", attribution.asn or "not established"),
        ("ISP / operator", attribution.isp or "not established"),
    ]

    # Never print a confident location above tier 1. At tier 2 the
    # location belongs to the infrastructure, not the sender; at tier 3 it
    # belongs to whatever the concealment layer chose to expose.
    if attribution.tier <= 1:
        rows.append(("Country", attribution.country or "not established"))
    else:
        rows.append(
            (
                "Country",
                f"{attribution.country or 'not established'} "
                "(location of the boundary host, NOT of the sender)",
            )
        )

    if attribution.is_vpn_or_tor:
        rows.append(("Anonymising layer", "yes, boundary sits on VPN, proxy or Tor infrastructure"))
    if attribution.notes:
        rows.append(("Notes", attribution.notes))

    caveats: list[str] = []
    if attribution.tier >= 3:
        caveats.append(
            "TIER 3: the origin of this message was deliberately concealed. No "
            "sender location or network can be stated. The concealment is itself "
            "the finding: ordinary correspondence does not arrive through an "
            "anonymising layer, and the effort taken to hide the origin is "
            "evidence about the sender's intent."
        )
    elif attribution.tier == 2:
        caveats.append(
            "TIER 2: attribution stops at the infrastructure. The boundary host and "
            "its operator are established; the person or machine behind them is not. "
            "Do not read the country above as the sender's location."
        )
    return rows, caveats


def _relay_rows(email: ParsedEmail) -> tuple[list[dict[str, Any]], Optional[int]]:
    """Received chain as printable rows, with the boundary index."""
    boundary = email.trust_boundary_index
    rows: list[dict[str, Any]] = []
    for hop in email.received_chain or []:
        rows.append(
            {
                "index": hop.index,
                "trusted": bool(hop.trusted),
                "label": "trusted" if hop.trusted else "attacker writable",
                "from_host": hop.from_host or "unknown",
                "from_ip": hop.from_ip or "unknown",
                "by_host": hop.by_host or "unknown",
                "timestamp": hop.timestamp.isoformat() if hop.timestamp else "no timestamp",
                "raw": hop.raw or "",
            }
        )
    return rows, boundary


def _campaign_block(
    email: ParsedEmail, verdict: Verdict, graph: Any = None
) -> Optional[dict[str, Any]]:
    """Campaign linkage, or None when this message is not in a campaign."""
    campaign_id = verdict.campaign_id
    summary: dict[str, Any] = {}
    if graph is not None and campaign_id:
        try:
            summary = graph.campaign_summary(campaign_id) or {}
        except Exception:
            summary = {}
    if not summary:
        summary = dict(email.meta.get("campaign_summary") or {})
    if not campaign_id and not summary:
        return None
    shared = summary.get("shared_artefacts") or summary.get("artefacts") or {}
    return {
        "campaign_id": campaign_id or summary.get("campaign_id") or "unknown",
        "message_count": summary.get("message_count"),
        "first_seen": summary.get("first_seen"),
        "last_seen": summary.get("last_seen"),
        "shared": shared,
    }


def _custody_block(
    email: ParsedEmail, verdict: Verdict, ledger: Optional[EvidenceLedger]
) -> dict[str, Any]:
    """Sealing hash, chain head and verification state."""
    block: dict[str, Any] = {
        "raw_sha256": email.raw_sha256,
        "seal_hash": None,
        "chain_head": verdict.evidence_hash or None,
        "entry_count": None,
        "intact": None,
        "broken_at": None,
        "sealed_at": None,
        "ledger_path": None,
    }
    if ledger is not None:
        try:
            report = ledger.verification_report()
            block.update(
                seal_hash=report.get("seal_hash"),
                chain_head=report.get("head") or block["chain_head"],
                entry_count=report.get("entry_count"),
                intact=report.get("intact"),
                broken_at=report.get("broken_at"),
                sealed_at=report.get("sealed_at"),
                ledger_path=report.get("path"),
            )
        except Exception:
            pass
    return block


def _report_model(
    email: ParsedEmail,
    verdict: Verdict,
    graph: Any = None,
    ledger: Optional[EvidenceLedger] = None,
) -> dict[str, Any]:
    """Everything both renderers need, assembled once."""
    case_id = str(email.meta.get("case_id") or case_id_for(email.raw_sha256))
    attribution_rows, caveats = _attribution_rows(email, verdict)
    relay_rows, boundary = _relay_rows(email)
    return {
        "case_id": case_id,
        "total_logit": sum(float(v) for v in verdict.contributions.values()),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_sha256": email.raw_sha256,
        "action": verdict.action,
        "colour": ACTION_COLOURS.get(verdict.action, "#333333"),
        "probability": verdict.probability,
        "verdict_class": verdict.verdict_class,
        "summary_rows": [
            ("From (display name)", email.from_display or "(none)"),
            ("From (address)", email.from_address or "(none)"),
            ("Reply-To", email.reply_to or "(not set)"),
            ("To", ", ".join(email.to_addresses or []) or "(none)"),
            ("Subject", email.subject or "(none)"),
            ("Date", email.date.isoformat() if email.date else "(no Date header)"),
            ("Message-ID", email.message_id or "(none)"),
        ],
        "evidence_rows": _evidence_rows(verdict),
        "arithmetic_rows": _arithmetic_rows(verdict),
        "attribution_rows": attribution_rows,
        "attribution_caveats": caveats,
        "relay_rows": relay_rows,
        "boundary_index": boundary,
        "campaign": _campaign_block(email, verdict, graph),
        "custody": _custody_block(email, verdict, ledger),
    }


# ----------------------------------------------------------------------
# PDF renderer
# ----------------------------------------------------------------------
def _pdf_styles() -> dict[str, Any]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "MGTitle", parent=base["Title"], fontSize=18, leading=22, spaceAfter=2
        ),
        "subtitle": ParagraphStyle(
            "MGSubtitle", parent=base["Normal"], fontSize=8.5, leading=11,
            textColor=colors.HexColor("#555555"),
        ),
        "section": ParagraphStyle(
            "MGSection", parent=base["Heading2"], fontSize=11, leading=13,
            spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#111111"),
        ),
        "body": ParagraphStyle(
            "MGBody", parent=base["Normal"], fontSize=8.5, leading=11, alignment=TA_LEFT
        ),
        "small": ParagraphStyle(
            "MGSmall", parent=base["Normal"], fontSize=7.5, leading=9.5,
            textColor=colors.HexColor("#444444"),
        ),
        "mono": ParagraphStyle(
            "MGMono", parent=base["Normal"], fontName="Courier", fontSize=6.5, leading=8
        ),
        "caveat": ParagraphStyle(
            "MGCaveat", parent=base["Normal"], fontSize=8.5, leading=11,
            textColor=colors.HexColor("#B00020"),
        ),
    }


def _kv_table(rows: list[tuple[str, str]], styles: dict[str, Any], width: float) -> Any:
    data = [
        [Paragraph(f"<b>{html.escape(str(key))}</b>", styles["small"]),
         Paragraph(html.escape(str(value)), styles["body"])]
        for key, value in rows
    ]
    table = Table(data, colWidths=[0.28 * width, 0.72 * width], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#DDDDDD")),
            ]
        )
    )
    return table


def _build_pdf(model: dict[str, Any], out_path: str) -> str:
    styles = _pdf_styles()
    page_width, _page_height = A4
    usable = page_width - 30 * mm

    def footer(canvas: Any, document: Any) -> None:
        """Page number and case id on every page."""
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(15 * mm, 10 * mm, f"{TITLE}  |  case {model['case_id']}")
        canvas.drawRightString(page_width - 15 * mm, 10 * mm, f"page {document.page}")
        canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
        canvas.line(15 * mm, 12.5 * mm, page_width - 15 * mm, 12.5 * mm)
        canvas.restoreState()

    story: list[Any] = []

    # 1. Header block -------------------------------------------------
    story.append(Paragraph(TITLE, styles["title"]))
    story.append(
        Paragraph(
            f"Case {html.escape(model['case_id'])} &nbsp;|&nbsp; generated "
            f"{html.escape(model['generated_at'])} (UTC)",
            styles["subtitle"],
        )
    )
    story.append(
        Paragraph(
            f"Raw message SHA-256: <font face='Courier'>{html.escape(model['raw_sha256'])}</font>",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#333333")))
    story.append(Spacer(1, 8))

    # 2. Verdict block ------------------------------------------------
    action_colour = colors.HexColor(model["colour"])
    verdict_cell = Paragraph(
        f"<font size='26' color='{model['colour']}'><b>{html.escape(model['action'])}</b></font>",
        styles["body"],
    )
    detail_cell = Paragraph(
        f"<b>Verdict class:</b> {html.escape(model['verdict_class'])}<br/>"
        f"<b>Probability of fraud:</b> {model['probability']:.4f}<br/>"
        f"<b>Total log-odds of fraud:</b> {model['total_logit']:+.2f} "
        f"(itemised in section 3)",
        styles["body"],
    )
    verdict_table = Table([[verdict_cell, detail_cell]], colWidths=[0.28 * usable, 0.72 * usable])
    verdict_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 1.0, action_colour),
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F7F7F7")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(verdict_table)

    # 3. Message summary ----------------------------------------------
    story.append(Paragraph("1. Message summary", styles["section"]))
    story.append(_kv_table(model["summary_rows"], styles, usable))

    # 4. Evidence table -----------------------------------------------
    story.append(Paragraph("2. Evidence table", styles["section"]))
    story.append(
        Paragraph(
            "One row per signal, ordered by contribution. Points are log-odds "
            "contributions to the verdict above: they sum, with the interaction "
            "terms and the base rate in section 3, to the total log-odds. A signal "
            "marked \"abstain, not scored\" could not run; that is missing data, not "
            "a clean result, and it still carries points, because the model scores "
            "an absence as an absence rather than as a zero.",
            styles["small"],
        )
    )
    story.append(Spacer(1, 4))

    header = ["", "Signal", "Score", "Status", "Points", "Evidence"]
    table_data: list[list[Any]] = [
        [Paragraph(f"<b>{html.escape(cell)}</b>", styles["small"]) for cell in header]
    ]
    for row in model["evidence_rows"]:
        score_text = row["score"]
        table_data.append(
            [
                Paragraph(html.escape(row["signal_id"]), styles["small"]),
                Paragraph(html.escape(row["name"]), styles["small"]),
                Paragraph(html.escape(score_text), styles["small"]),
                Paragraph(html.escape(row["status"]), styles["small"]),
                Paragraph(html.escape(row["points_text"]), styles["small"]),
                Paragraph(html.escape(row["evidence"]), styles["small"]),
            ]
        )
    widths = [
        0.05 * usable, 0.15 * usable, 0.12 * usable,
        0.09 * usable, 0.08 * usable, 0.51 * usable,
    ]
    evidence_table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    evidence_style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#999999")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for position, row in enumerate(model["evidence_rows"], start=1):
        if row["abstained"]:
            evidence_style.append(
                ("BACKGROUND", (0, position), (-1, position), colors.HexColor("#FAF3E0"))
            )
            evidence_style.append(
                ("TEXTCOLOR", (2, position), (3, position), colors.HexColor("#8A5A00"))
            )
        elif row["points"] >= 1.0:
            evidence_style.append(
                ("TEXTCOLOR", (4, position), (4, position), colors.HexColor("#B00020"))
            )
    evidence_table.setStyle(TableStyle(evidence_style))
    story.append(evidence_table)

    # Model arithmetic, so the numbers visibly add up.
    story.append(Paragraph("3. Model arithmetic", styles["section"]))
    story.append(_kv_table(model["arithmetic_rows"], styles, usable))

    # 5. Origin attribution -------------------------------------------
    story.append(Paragraph("4. Origin attribution", styles["section"]))
    for caveat in model["attribution_caveats"]:
        story.append(Paragraph(html.escape(caveat), styles["caveat"]))
        story.append(Spacer(1, 4))
    story.append(_kv_table(model["attribution_rows"], styles, usable))

    # 6. Relay trace ---------------------------------------------------
    story.append(Paragraph("5. Relay trace", styles["section"]))
    boundary = model["boundary_index"]
    if boundary is None:
        story.append(
            Paragraph(
                "No trust boundary was established for this chain, so every hop "
                "below must be treated as attacker writable.",
                styles["small"],
            )
        )
    else:
        story.append(
            Paragraph(
                f"Hops 0 to {boundary} were recorded by infrastructure under our "
                "control and can be relied on. Everything below the rule was "
                "written by hosts we do not control and may be fabricated.",
                styles["small"],
            )
        )
    story.append(Spacer(1, 4))
    if not model["relay_rows"]:
        story.append(Paragraph("No Received headers were present.", styles["body"]))
    for row in model["relay_rows"]:
        colour = "#1B7A3D" if row["trusted"] else "#B00020"
        story.append(
            KeepTogether(
                [
                    Paragraph(
                        f"<b>hop {row['index']}</b> &nbsp; "
                        f"<font color='{colour}'><b>{html.escape(row['label'])}</b></font> "
                        f"&nbsp; from {html.escape(row['from_host'])} "
                        f"[{html.escape(row['from_ip'])}] "
                        f"by {html.escape(row['by_host'])} &nbsp; "
                        f"{html.escape(row['timestamp'])}",
                        styles["small"],
                    ),
                    Paragraph(html.escape(row["raw"]), styles["mono"]),
                    Spacer(1, 3),
                ]
            )
        )
        if boundary is not None and row["index"] == boundary:
            story.append(Spacer(1, 2))
            story.append(
                HRFlowable(width="100%", thickness=1.4, color=colors.HexColor("#B00020"))
            )
            story.append(
                Paragraph(
                    "<b>TRUST BOUNDARY</b> - everything below this line is "
                    "attacker writable",
                    styles["caveat"],
                )
            )
            story.append(Spacer(1, 4))

    # 7. Campaign linkage ---------------------------------------------
    campaign = model["campaign"]
    if campaign:
        story.append(Paragraph("6. Campaign linkage", styles["section"]))
        rows: list[tuple[str, str]] = [("Campaign id", str(campaign["campaign_id"]))]
        if campaign["message_count"] is not None:
            rows.append(("Messages in campaign", str(campaign["message_count"])))
        if campaign["first_seen"]:
            rows.append(("First seen", str(campaign["first_seen"])))
        if campaign["last_seen"]:
            rows.append(("Last seen", str(campaign["last_seen"])))
        for kind, values in sorted((campaign["shared"] or {}).items()):
            listed = values if isinstance(values, (list, tuple)) else [values]
            rows.append((f"Shared {kind.replace('_', ' ')}", ", ".join(str(v) for v in listed)))
        story.append(_kv_table(rows, styles, usable))

    # 8. Chain of custody ---------------------------------------------
    story.append(Paragraph("7. Chain of custody", styles["section"]))
    custody = model["custody"]
    custody_rows: list[tuple[str, str]] = [
        ("Raw message SHA-256", custody["raw_sha256"] or "not recorded"),
        ("Sealing entry hash", custody["seal_hash"] or "not recorded in a ledger"),
        ("Current chain head", custody["chain_head"] or GENESIS_HASH),
    ]
    if custody["sealed_at"]:
        custody_rows.append(("Sealed at", str(custody["sealed_at"])))
    if custody["entry_count"] is not None:
        custody_rows.append(("Ledger entries", str(custody["entry_count"])))
    if custody["intact"] is not None:
        state = "intact, every link verified" if custody["intact"] else (
            f"BROKEN at entry {custody['broken_at']}"
        )
        custody_rows.append(("Chain verification", state))
    if custody["ledger_path"]:
        custody_rows.append(("Ledger", str(custody["ledger_path"])))
    story.append(_kv_table(custody_rows, styles, usable))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Each ledger entry stores the SHA-256 of the entry before it. Altering "
            "any earlier record changes its hash and therefore breaks every hash "
            "after it, so the chain cannot be edited silently; the first broken "
            "link is reported by index. The sealing entry records the hash of the "
            "message as received, taken before any parsing, which is what makes it "
            "evidence about the message rather than about this tool.",
            styles["small"],
        )
    )

    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    document = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title=f"{TITLE} - {model['case_id']}",
        author="MailGuard AI",
        subject=f"{model['action']} / {model['verdict_class']}",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path


# ----------------------------------------------------------------------
# HTML renderer, used when reportlab is not installed
# ----------------------------------------------------------------------
def _html_kv(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
    )
    return f"<table class='kv'>{body}</table>"


def _build_html(model: dict[str, Any], out_path: str) -> str:
    parts: list[str] = []
    parts.append(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(TITLE)} - {html.escape(model['case_id'])}</title><style>"
        "body{font-family:Helvetica,Arial,sans-serif;font-size:13px;color:#111;"
        "max-width:980px;margin:24px auto;padding:0 16px;line-height:1.45}"
        "h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 6px;"
        "border-bottom:1px solid #ddd;padding-bottom:3px}"
        ".sub{color:#555;font-size:11px}.mono{font-family:Courier,monospace;font-size:11px}"
        "table{border-collapse:collapse;width:100%;margin:6px 0}"
        "table.kv th{text-align:left;width:26%;vertical-align:top;font-weight:600;"
        "padding:3px 8px 3px 0;border-bottom:1px solid #eee}"
        "table.kv td{vertical-align:top;padding:3px 0;border-bottom:1px solid #eee}"
        "table.ev th,table.ev td{border:1px solid #ddd;padding:4px 6px;"
        "vertical-align:top;font-size:11.5px;text-align:left}"
        "table.ev th{background:#eee}tr.abstain{background:#faf3e0;color:#8a5a00}"
        ".verdict{display:flex;align-items:center;gap:18px;border:1px solid #333;"
        "padding:12px 14px;margin:10px 0}.action{font-size:30px;font-weight:700}"
        ".caveat{color:#B00020;font-weight:600}.boundary{border:0;border-top:2px solid #B00020;"
        "margin:8px 0 2px}.hop{margin:6px 0}.trusted{color:#1B7A3D;font-weight:600}"
        ".untrusted{color:#B00020;font-weight:600}.note{color:#444;font-size:11.5px}"
        "footer{margin-top:28px;color:#666;font-size:11px;border-top:1px solid #ddd;padding-top:6px}"
        "</style></head><body>"
    )
    # 1. Header
    parts.append(f"<h1>{html.escape(TITLE)}</h1>")
    parts.append(
        f"<div class='sub'>Case {html.escape(model['case_id'])} &middot; generated "
        f"{html.escape(model['generated_at'])} (UTC)</div>"
    )
    parts.append(
        f"<div class='sub'>Raw message SHA-256: <span class='mono'>"
        f"{html.escape(model['raw_sha256'])}</span></div>"
    )
    # 2. Verdict
    parts.append(
        f"<div class='verdict' style='border-color:{model['colour']}'>"
        f"<div class='action' style='color:{model['colour']}'>{html.escape(model['action'])}</div>"
        f"<div><b>Verdict class:</b> {html.escape(model['verdict_class'])}<br>"
        f"<b>Probability of fraud:</b> {model['probability']:.4f}<br>"
        f"<b>Total log-odds of fraud:</b> {model['total_logit']:+.2f} "
        "(itemised in section 3)</div></div>"
    )
    # 3. Message summary
    parts.append("<h2>1. Message summary</h2>")
    parts.append(_html_kv(model["summary_rows"]))
    # 4. Evidence table
    parts.append("<h2>2. Evidence table</h2>")
    parts.append(
        "<div class='note'>One row per signal, ordered by contribution. Points are "
        "log-odds contributions to the verdict above. A signal marked "
        "\"abstain, not scored\" could not run; that is missing data, not a clean "
        "result, and it still carries points, because the model scores an absence "
        "as an absence rather than as a zero.</div>"
    )
    parts.append(
        "<table class='ev'><tr><th></th><th>Signal</th><th>Score</th><th>Status</th>"
        "<th>Points</th><th>Evidence</th></tr>"
    )
    for row in model["evidence_rows"]:
        css = " class='abstain'" if row["abstained"] else ""
        parts.append(
            f"<tr{css}><td>{html.escape(row['signal_id'])}</td>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td>{html.escape(row['score'])}</td>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td>{html.escape(row['points_text'])}</td>"
            f"<td>{html.escape(row['evidence'])}</td></tr>"
        )
    parts.append("</table>")
    parts.append("<h2>3. Model arithmetic</h2>")
    parts.append(_html_kv(model["arithmetic_rows"]))
    # 5. Attribution
    parts.append("<h2>4. Origin attribution</h2>")
    for caveat in model["attribution_caveats"]:
        parts.append(f"<p class='caveat'>{html.escape(caveat)}</p>")
    parts.append(_html_kv(model["attribution_rows"]))
    # 6. Relay trace
    parts.append("<h2>5. Relay trace</h2>")
    boundary = model["boundary_index"]
    if boundary is None:
        parts.append(
            "<div class='note'>No trust boundary was established, so every hop below "
            "must be treated as attacker writable.</div>"
        )
    else:
        parts.append(
            f"<div class='note'>Hops 0 to {boundary} were recorded by infrastructure "
            "under our control. Everything below the rule was written by hosts we do "
            "not control and may be fabricated.</div>"
        )
    for row in model["relay_rows"]:
        css = "trusted" if row["trusted"] else "untrusted"
        parts.append(
            f"<div class='hop'><b>hop {row['index']}</b> "
            f"<span class='{css}'>{html.escape(row['label'])}</span> "
            f"from {html.escape(row['from_host'])} [{html.escape(row['from_ip'])}] "
            f"by {html.escape(row['by_host'])} &middot; {html.escape(row['timestamp'])}"
            f"<div class='mono'>{html.escape(row['raw'])}</div></div>"
        )
        if boundary is not None and row["index"] == boundary:
            parts.append("<hr class='boundary'>")
            parts.append(
                "<div class='caveat'>TRUST BOUNDARY - everything below this line is "
                "attacker writable</div>"
            )
    # 7. Campaign
    campaign = model["campaign"]
    if campaign:
        parts.append("<h2>6. Campaign linkage</h2>")
        rows: list[tuple[str, str]] = [("Campaign id", str(campaign["campaign_id"]))]
        if campaign["message_count"] is not None:
            rows.append(("Messages in campaign", str(campaign["message_count"])))
        if campaign["first_seen"]:
            rows.append(("First seen", str(campaign["first_seen"])))
        if campaign["last_seen"]:
            rows.append(("Last seen", str(campaign["last_seen"])))
        for kind, values in sorted((campaign["shared"] or {}).items()):
            listed = values if isinstance(values, (list, tuple)) else [values]
            rows.append((f"Shared {kind.replace('_', ' ')}", ", ".join(str(v) for v in listed)))
        parts.append(_html_kv(rows))
    # 8. Custody
    parts.append("<h2>7. Chain of custody</h2>")
    custody = model["custody"]
    custody_rows: list[tuple[str, str]] = [
        ("Raw message SHA-256", custody["raw_sha256"] or "not recorded"),
        ("Sealing entry hash", custody["seal_hash"] or "not recorded in a ledger"),
        ("Current chain head", custody["chain_head"] or GENESIS_HASH),
    ]
    if custody["sealed_at"]:
        custody_rows.append(("Sealed at", str(custody["sealed_at"])))
    if custody["entry_count"] is not None:
        custody_rows.append(("Ledger entries", str(custody["entry_count"])))
    if custody["intact"] is not None:
        custody_rows.append(
            (
                "Chain verification",
                "intact, every link verified"
                if custody["intact"]
                else f"BROKEN at entry {custody['broken_at']}",
            )
        )
    if custody["ledger_path"]:
        custody_rows.append(("Ledger", str(custody["ledger_path"])))
    parts.append(_html_kv(custody_rows))
    parts.append(
        "<div class='note'>Each ledger entry stores the SHA-256 of the entry before "
        "it. Altering any earlier record changes its hash and therefore breaks every "
        "hash after it, so the chain cannot be edited silently; the first broken link "
        "is reported by index. The sealing entry records the hash of the message as "
        "received, taken before any parsing.</div>"
    )
    parts.append(
        f"<footer>{html.escape(TITLE)} &middot; case {html.escape(model['case_id'])} "
        "&middot; rendered as HTML because reportlab is not installed; "
        "install reportlab for the PDF.</footer></body></html>"
    )

    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    return out_path


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_report(
    email: ParsedEmail,
    verdict: Verdict,
    out_path: str,
    graph: Any = None,
    ledger: Optional[EvidenceLedger] = None,
) -> str:
    """Write the forensic report and return the path actually written.

    Sections, in order: header, verdict, message summary, evidence table,
    model arithmetic, origin attribution, relay trace, campaign linkage
    (omitted when there is no campaign), chain of custody.

    `graph` and `ledger` are optional. Given a CampaignGraph the campaign
    section prints real counts and shared artefacts; given an
    EvidenceLedger the custody section prints the verified chain state.
    Without them the report still builds from what the Verdict and the
    email's meta carry, and says what it does not know.

    If reportlab is missing, an equivalent HTML report is written at the
    same path with a .html extension and that path is returned, so this
    function never simply fails.
    """
    model = _report_model(email, verdict, graph=graph, ledger=ledger)
    if HAS_REPORTLAB:
        target = out_path if out_path.lower().endswith(".pdf") else out_path + ".pdf"
        return _build_pdf(model, target)
    base = out_path[:-4] if out_path.lower().endswith(".pdf") else out_path
    return _build_html(model, base + ".html")


if __name__ == "__main__":  # pragma: no cover - demonstration
    import tempfile

    from tests.factory import make_email, make_signals, make_verdict

    print("MailGuard forensic report demonstration")
    print("=" * 78)
    print(f"reportlab installed: {HAS_REPORTLAB}")

    demo_email = make_email()
    demo_verdict = make_verdict(demo_email, make_signals())

    # A ledger, so the custody section prints a verified chain rather than
    # "not recorded". Written to a temporary file: the demo must not
    # append to whatever real ledger the host is using.
    ledger_dir = tempfile.mkdtemp(prefix="mailguard-report-demo-")
    demo_ledger = EvidenceLedger(os.path.join(ledger_dir, "demo_ledger.jsonl"))
    demo_ledger.seal_raw(
        demo_email.raw_sha256,
        source=str(demo_email.meta.get("ingest_source", "smtp://unknown")),
        operator=str(demo_email.meta.get("operator", "demo")),
    )
    case = case_id_for(demo_email.raw_sha256)
    demo_ledger.append({"type": "parsed", "case_id": case, "hops": len(demo_email.received_chain)})
    demo_ledger.append(
        {
            "type": "signals",
            "case_id": case,
            "scores": {s.signal_id: (s.status if s.status != "ok" else s.score) for s in demo_verdict.signals},
        }
    )
    demo_ledger.append(
        {
            "type": "verdict",
            "case_id": case,
            "probability": demo_verdict.probability,
            "action": demo_verdict.action,
            "verdict_class": demo_verdict.verdict_class,
            "contributions": demo_verdict.contributions,
        }
    )

    # A campaign graph with a second, related message in it, so the
    # campaign section has something real to report.
    from mailguard.graph.campaign_graph import CampaignGraph

    demo_graph = CampaignGraph(backend="memory")
    demo_verdict.campaign_id = demo_graph.add_verdict(demo_email, demo_verdict)
    sibling = make_email(
        message_id="<20260918041955.44902.0a71@axis-alerts.example>",
        from_display="Axis Bank Alerts",
        from_address="alerts@axis-alerts.example",
        from_domain="axis-alerts.example",
        raw_sha256="b" * 64,
    )
    demo_graph.add_verdict(sibling, make_verdict(sibling, make_signals()))
    demo_verdict.campaign_id = demo_graph.find_campaign(demo_email) or demo_verdict.campaign_id

    written = build_report(
        demo_email,
        demo_verdict,
        "sample_report.pdf",
        graph=demo_graph,
        ledger=demo_ledger,
    )
    size = os.path.getsize(written)
    print(f"verdict            : {demo_verdict.action} / {demo_verdict.verdict_class} "
          f"p={demo_verdict.probability:.4f}")
    print(f"campaign           : {demo_verdict.campaign_id} "
          f"({demo_graph.campaign_summary(demo_verdict.campaign_id)['message_count']} messages)")
    intact, broken = demo_ledger.verify_chain()
    print(f"ledger             : {demo_ledger.count()} entries, intact={intact}, broken_at={broken}")
    print(f"report written     : {written} ({size} bytes)")
    if not HAS_REPORTLAB:
        print("note               : reportlab is not installed, so the HTML report was")
        print("                     written instead. `pip install reportlab` for the PDF.")
