"""Command line entry point: run the MailGuard pipeline as far as it currently exists.

    python -m mailguard.cli --eml samples/cousin_domain_phish.eml
    python -m mailguard.cli --eml <file> --json
    python -m mailguard.cli --stdin
    python -m mailguard.cli --imap-host imap.example.org --imap-user analyst --imap-limit 5
    python -m mailguard.cli --eml <file> --report case.pdf --trace-map case.html

Flow:

  1. load the raw bytes, hash them, and seal that hash into the evidence
     ledger immediately, before any parsing
  2. build the ParsedEmail
  3. annotate the trust boundary and run attribution
  4. discover every signal module and run them in parallel, each inside
     SIGNAL_TIMEOUT_MS; an overrunning signal is recorded as abstain
  5. fuse, if the fusion module exists
  6. correlate a WARN or BLOCK verdict into the campaign graph
  7. hash chain every stage (parse, signals, verdict, report) into the
     ledger; the verdict carries the chain head as its evidence hash
  8. build the PDF report and the trace map, if asked for
  9. print the signal table, the verdict, the campaign, the attribution
     tier and the trust boundary

The ML half of the project (signals x2 / x4 / x5 / x7, fusion, the report)
is built independently. This CLI works with all of it, some of it, or
none of it: missing signals are simply not discovered, and a missing
fusion or report module is reported in one line instead of crashing.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Optional, Sequence

from mailguard.core.config import Config, load_config, set_active_config
from mailguard.core.models import ParsedEmail, SignalResult, Verdict
from mailguard.core.privacy import maybe_mask
from mailguard.forensics.attribution import attribute
from mailguard.forensics.trust_boundary import annotate_chain, describe_boundary
from mailguard.ingest.loader import load_eml_file, load_from_imap, load_from_stdin, sha256_of
from mailguard.parsing.url_extractor import build_parsed_email
from mailguard.signals import DISCOVERY_ERRORS, discover_signals

RULE = "-" * 100


def _out(line: str = "") -> None:
    """Print one line, replacing characters the console cannot encode."""
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _clip(text: Any, width: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= width else value[: max(0, width - 3)] + "..."


# ----------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------
def _abstain(module: Any, reason: str) -> SignalResult:
    return SignalResult(
        signal_id=str(getattr(module, "SIGNAL_ID", "x?")),
        name=str(getattr(module, "SIGNAL_NAME", getattr(module, "__name__", "unknown"))),
        score=0.0,
        status="abstain",
        evidence_row=reason,
    )


def run_signals(email: ParsedEmail, modules: list, timeout_ms: int) -> list[SignalResult]:
    """Run every signal in parallel; any signal still running after timeout_ms abstains."""
    if not modules:
        return []
    results: dict[int, SignalResult] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(modules), thread_name_prefix="signal")
    futures = {executor.submit(module.run, email): i for i, module in enumerate(modules)}
    done, _pending = concurrent.futures.wait(futures, timeout=max(0.001, timeout_ms / 1000.0))
    for future, i in futures.items():
        module = modules[i]
        if future in done:
            try:
                result = future.result()
                results[i] = result if isinstance(result, SignalResult) else _abstain(module, "signal returned no SignalResult")
            except Exception as exc:  # a signal broke its never-raise contract
                results[i] = _abstain(module, f"signal raised {type(exc).__name__}")
        else:
            results[i] = _abstain(module, f"exceeded the {timeout_ms} ms signal time budget")
    # Do not wait for overrunning threads; their results are discarded.
    executor.shutdown(wait=False, cancel_futures=True)
    return [results[i] for i in range(len(modules))]


class Custody:
    """Writes each pipeline stage into the evidence ledger, and never breaks the run.

    A ledger that cannot be written (missing module, read-only disk) costs
    the chain of custody, which is reported in one line, not the verdict.
    """

    def __init__(self, ledger: Any, case_id: str) -> None:
        self.ledger = ledger
        self.case_id = case_id
        self.head: Optional[str] = None
        self.note = "" if ledger is not None else "evidence ledger disabled"

    def seal(self, raw_sha256: str, source: str) -> None:
        """The first record for a message: its hash as received, before parsing."""
        if self.ledger is None:
            return
        operator = os.environ.get("MAILGUARD_OPERATOR") or _whoami()
        try:
            self.head = self.ledger.seal_raw(raw_sha256, source, operator)
        except ValueError:
            # Already sealed by an earlier run. Same bytes means the same
            # case id, so the original seal still stands; record the repeat.
            self.append("reanalysis", raw_sha256=raw_sha256, source=source, operator=operator)
        except Exception as exc:
            self.note = f"ledger unavailable: {type(exc).__name__}: {exc}"
            self.ledger = None

    def append(self, entry_type: str, **payload: Any) -> None:
        if self.ledger is None:
            return
        try:
            self.head = self.ledger.append({"type": entry_type, "case_id": self.case_id, **payload})
        except Exception as exc:
            self.note = f"ledger write failed: {type(exc).__name__}: {exc}"
            self.ledger = None


def _whoami() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _case_id(raw_sha256: str) -> str:
    try:
        from mailguard.evidence.ledger import case_id_for
    except ImportError:
        return f"MG-{raw_sha256[:12].upper()}"
    return case_id_for(raw_sha256)


def _file_sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def analyse(
    raw: bytes,
    source: str,
    config: Config,
    report_path: Optional[str] = None,
    ledger: Any = None,
    graph: Any = None,
    trace_map_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run the whole pipeline on one message and return everything it produced."""
    raw_sha256 = sha256_of(raw)                        # 1. hash first, before any parsing
    case_id = _case_id(raw_sha256)
    custody = Custody(ledger, case_id)
    custody.seal(raw_sha256, source)                   #    and seal it, still before parsing

    email = build_parsed_email(raw, source)            # 2. parse
    email.meta["case_id"] = case_id
    annotate_chain(email, config.trusted_hosts)        # 3. trust boundary
    email.attribution = attribute(email, config)       #    and attribution
    custody.append(
        "parsed",
        message_id=email.message_id,
        hops=len(email.received_chain),
        trust_boundary_index=email.trust_boundary_index,
        boundary_ip=email.attribution.boundary_ip if email.attribution else None,
        tier=email.attribution.tier if email.attribution else None,
        url_count=len(email.urls),
        attachment_sha256=[a.sha256 for a in email.attachments],
    )

    modules = discover_signals()                       # 4. whatever signals exist
    started = time.perf_counter()
    signals = run_signals(email, modules, config.signal_timeout_ms)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    custody.append("signals", scores={s.signal_id: (s.score if s.status == "ok" else s.status) for s in signals})

    verdict: Optional[Verdict] = None                  # 5. fusion, if it exists
    fusion_note = ""
    try:
        from mailguard.fusion.ebm_fusion import fuse
        verdict = fuse(signals, email)
    except ImportError:
        verdict = None
        fusion_note = "fusion is not available yet (mailguard.fusion.ebm_fusion not installed)"
    except Exception as exc:
        verdict = None
        fusion_note = f"fusion failed: {type(exc).__name__}: {exc}"

    # 6. campaign correlation. Only mail that is at least suspicious joins
    # a campaign: a campaign is a grouping of fraud for investigators, and
    # filing every legitimate newsletter into one would bury them.
    campaign_note = ""
    if verdict is not None and graph is not None and verdict.action != "PASS":
        try:
            graph.add_verdict(email, verdict)
            email.meta["campaign_summary"] = graph.campaign_summary(verdict.campaign_id)
        except Exception as exc:
            campaign_note = f"campaign graph failed: {type(exc).__name__}: {exc}"

    # 7. the verdict into the chain; its head becomes the evidence hash
    if verdict is not None:
        custody.append(
            "verdict",
            probability=verdict.probability,
            action=verdict.action,
            verdict_class=verdict.verdict_class,
            contributions=verdict.contributions,
            campaign_id=verdict.campaign_id,
            fusion=email.meta.get("fusion"),
        )
        verdict.evidence_hash = custody.head

    try:                                               # 8. report, if it exists
        from mailguard.evidence.pdf_report import build_report
    except ImportError:
        build_report = None
    report_note = ""
    written: Optional[str] = None
    if report_path:
        if build_report is None:
            report_note = "PDF report is not available yet (mailguard.evidence.pdf_report not installed)"
        elif verdict is None:
            report_note = "PDF report needs a verdict, and fusion is not available"
        else:
            try:
                written = build_report(email, verdict, report_path, graph=graph, ledger=custody.ledger)
                custody.append("report", path=os.path.abspath(written), report_sha256=_file_sha256(written))
            except Exception as exc:
                report_note = f"report failed: {type(exc).__name__}: {exc}"

    map_written: Optional[str] = None
    if trace_map_path:
        try:
            from mailguard.evidence.trace_map import build_trace_map

            map_written = build_trace_map(email, verdict, trace_map_path)
        except Exception as exc:
            report_note = (report_note + "; " if report_note else "") + f"trace map failed: {type(exc).__name__}: {exc}"

    return {
        "raw_sha256": raw_sha256,
        "case_id": case_id,
        "email": email,
        "signals": signals,
        "verdict": verdict,
        "fusion_note": fusion_note,
        "campaign_note": campaign_note,
        "ledger_head": custody.head,
        "ledger_note": custody.note,
        "report_path": written,
        "trace_map_path": map_written,
        "report_note": report_note,
        "elapsed_ms": elapsed_ms,
        "discovered": [getattr(m, "SIGNAL_ID", "?") for m in modules],
        "discovery_errors": dict(DISCOVERY_ERRORS),
    }


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def print_result(result: dict[str, Any]) -> None:
    email: ParsedEmail = result["email"]
    verdict: Optional[Verdict] = result["verdict"]
    _out(RULE)
    _out(f"MailGuard AI   {email.meta.get('source', '')}")
    _out(RULE)
    _out(f"sha256   {result['raw_sha256']}  (taken before parsing)")
    _out(f"case     {result['case_id']}")
    _out(f"from     {_clip(maybe_mask(email.from_display), 40)} <{maybe_mask(email.from_address)}>")
    _out(f"subject  {_clip(maybe_mask(email.subject), 80)}")
    _out("")

    contributions = verdict.contributions if verdict else {}
    _out(f"{'id':<4} {'signal':<26} {'score':>6}  {'status':<8} {'points':>7}  evidence")
    _out("-" * 100)
    for signal in result["signals"]:
        points = contributions.get(signal.signal_id)
        points_text = f"{points:+.2f}" if isinstance(points, (int, float)) else ""
        score = f"{signal.score:.2f}" if signal.status == "ok" else "-"
        _out(f"{signal.signal_id:<4} {_clip(signal.name, 26):<26} {score:>6}  {_clip(signal.status, 8):<8} "
             f"{points_text:>7}  {_clip(maybe_mask(signal.evidence_row), 90)}")
    if not result["signals"]:
        _out("(no signal modules found)")
    for module, error in result["discovery_errors"].items():
        _out(f"     skipped {module}: {_clip(error, 70)}")
    _out(f"     {len(result['signals'])} signals in {result['elapsed_ms']} ms")
    _out("")

    if verdict is not None:
        _out(f"VERDICT  {verdict.action}   {verdict.verdict_class}   p={verdict.probability:.4f}")
        if verdict.campaign_id:
            summary = email.meta.get("campaign_summary") or {}
            count = summary.get("message_count")
            _out(f"campaign {verdict.campaign_id}" + (f"  ({count} messages)" if count else ""))
    else:
        _out(f"VERDICT  none: {result['fusion_note']}")
    if result.get("campaign_note"):
        _out(f"campaign {result['campaign_note']}")
    if result.get("ledger_head"):
        _out(f"ledger   chain head {result['ledger_head']}")
    elif result.get("ledger_note"):
        _out(f"ledger   {result['ledger_note']}")
    _out("")

    attribution = email.attribution
    if attribution is not None:
        network = ", ".join(x for x in (attribution.asn, attribution.isp, attribution.city, attribution.country) if x)
        _out(f"origin   tier {attribution.tier} ({attribution.tier_name}), boundary IP "
             f"{attribution.boundary_ip or 'none'}" + (f", {network}" if network else ""))
        if attribution.notes:
            _out(f"         {_clip(attribution.notes, 200)}")
    _out(f"boundary {describe_boundary(email)}")
    for hop in email.received_chain:
        label = "trusted   " if hop.trusted else "UNVERIFIED"
        if email.trust_boundary_index is not None and hop.index == email.trust_boundary_index:
            _out("   ---- trust boundary: everything below this line is attacker writable ----")
        _out(f"   [{hop.index}] {label} by {_clip(hop.by_host or '?', 32):<32} from {_clip(hop.from_host or '?', 30)} "
             f"[{hop.from_ip or 'no IP'}]")
    if email.received_chain and email.trust_boundary_index is not None \
            and email.trust_boundary_index >= len(email.received_chain):
        _out("   ---- trust boundary: the sender connected to the hop above directly ----")

    if result["report_path"]:
        _out(f"report   {result['report_path']}")
    if result.get("trace_map_path"):
        _out(f"map      {result['trace_map_path']}")
    if result["report_note"]:
        _out(f"report   {result['report_note']}")


def to_json(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe view of one result. Raw bytes and attachment payloads are left out."""
    email: ParsedEmail = result["email"]
    verdict: Optional[Verdict] = result["verdict"]
    signals = []
    for signal in result["signals"]:
        row = asdict(signal)
        row["evidence_row"] = maybe_mask(row["evidence_row"])
        signals.append(row)
    return {
        "raw_sha256": result["raw_sha256"],
        "case_id": result["case_id"],
        "source": email.meta.get("source"),
        "message_id": email.message_id,
        "from": {"display": maybe_mask(email.from_display), "address": maybe_mask(email.from_address),
                 "domain": email.from_domain},
        "subject": maybe_mask(email.subject),
        "signals": signals,
        "verdict": None if verdict is None else {
            "action": verdict.action,
            "verdict_class": verdict.verdict_class,
            "probability": verdict.probability,
            "contributions": verdict.contributions,
            "campaign_id": verdict.campaign_id,
            "evidence_hash": verdict.evidence_hash,
        },
        "fusion": email.meta.get("fusion"),
        "fusion_note": result["fusion_note"] or None,
        "campaign": email.meta.get("campaign_summary"),
        "ledger_note": result.get("ledger_note") or None,
        "trust_boundary": {
            "index": email.trust_boundary_index,
            "description": describe_boundary(email),
            "hops": [
                {"index": h.index, "trusted": h.trusted, "by": h.by_host, "from": h.from_host,
                 "ip": h.from_ip, "timestamp": h.timestamp}
                for h in email.received_chain
            ],
        },
        "attribution": asdict(email.attribution) if email.attribution else None,
        "urls": [asdict(u) for u in email.urls],
        "attachments": [
            {"filename": a.filename, "content_type": a.content_type, "magic_type": a.magic_type,
             "size_bytes": a.size_bytes, "sha256": a.sha256}
            for a in email.attachments
        ],
        "report_path": result["report_path"],
        "trace_map_path": result.get("trace_map_path"),
        "report_note": result["report_note"] or None,
        "discovery_errors": result["discovery_errors"],
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mailguard.cli",
        description="MailGuard AI: score an email with seven signals and trace its sender honestly.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eml", help="path to a .eml file")
    source.add_argument("--stdin", action="store_true", help="read one raw message from standard input")
    source.add_argument("--imap-host", help="IMAP server (TLS); password from MAILGUARD_IMAP_PASSWORD")
    parser.add_argument("--imap-user", help="IMAP user name")
    parser.add_argument("--imap-mailbox", default="INBOX", help="IMAP mailbox (default INBOX)")
    parser.add_argument("--imap-limit", type=int, default=10, help="most recent N messages (default 10)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--report", help="also write the forensic PDF report to this path")
    parser.add_argument("--trace-map", help="also write the HTML trace map to this path")
    parser.add_argument("--ledger", help="evidence ledger file (default MAILGUARD_LEDGER_PATH or "
                                         "mailguard_ledger.jsonl)")
    parser.add_argument("--no-ledger", action="store_true", help="do not write to the evidence ledger")
    parser.add_argument("--graph", help="persist the campaign graph to this JSON file (default "
                                        "MAILGUARD_GRAPH_PATH; in memory for this run only when unset)")
    parser.add_argument("--mask", action="store_true", help="mask personal data in all output")
    parser.add_argument("--trusted-host", action="append", metavar="HOST",
                        help="a mail server we control; repeatable; replaces TRUSTED_HOSTS")
    parser.add_argument("--offline", action="store_true", help="disable every network lookup")
    parser.add_argument("--timeout-ms", type=int, help="per signal time budget (default SIGNAL_TIMEOUT_MS)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show warnings from signal discovery")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(
        trusted_hosts=args.trusted_host or None,
        network_lookups=False if args.offline else None,
        signal_timeout_ms=args.timeout_ms,
        mask_pii=True if args.mask else None,
    )
    set_active_config(config)

    ledger = None
    if not args.no_ledger:
        try:
            from mailguard.evidence.ledger import DEFAULT_LEDGER_PATH, EvidenceLedger

            ledger = EvidenceLedger(args.ledger or DEFAULT_LEDGER_PATH)
        except ImportError:
            ledger = None
    graph = None
    try:
        from mailguard.graph.campaign_graph import CampaignGraph

        graph = CampaignGraph(store_path=args.graph or "")
        graph.prune(config.retention_days)
    except ImportError:
        graph = None
    except Exception as exc:
        _out(f"warning: campaign graph unavailable: {type(exc).__name__}: {exc}")
        graph = None

    messages: list[tuple[bytes, str]] = []
    try:
        if args.eml:
            messages.append(load_eml_file(args.eml))
        elif args.stdin:
            messages.append(load_from_stdin())
        else:
            if not args.imap_user:
                _out("error: --imap-host needs --imap-user")
                return 2
            password = os.environ.get("MAILGUARD_IMAP_PASSWORD", "")
            messages.extend(load_from_imap(args.imap_host, args.imap_user, password, args.imap_mailbox, args.imap_limit))
            if not messages:
                _out("no messages fetched from IMAP (see warnings with -v)")
                return 1
    except (OSError, ValueError) as exc:
        _out(f"error: could not read input: {exc}")
        return 1

    outputs: list[dict[str, Any]] = []
    for index, (raw, source) in enumerate(messages):
        report = args.report
        trace_map = args.trace_map
        if len(messages) > 1:
            if report:
                base, ext = os.path.splitext(report)
                report = f"{base}-{index + 1}{ext or '.pdf'}"
            if trace_map:
                base, ext = os.path.splitext(trace_map)
                trace_map = f"{base}-{index + 1}{ext or '.html'}"
        result = analyse(raw, source, config, report_path=report, ledger=ledger, graph=graph,
                         trace_map_path=trace_map)
        if args.json:
            outputs.append(to_json(result))
        else:
            print_result(result)
    if args.json:
        _out(json.dumps(outputs[0] if len(outputs) == 1 else outputs, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
