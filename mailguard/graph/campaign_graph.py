"""Campaign graph: linking separate attacks that share hidden infrastructure.

One fraudulent mail is an incident. Forty mails sharing a Reply-To
address, a boundary IP and a DKIM selector are a campaign, and that is a
different conversation with a different value: it tells an investigator
the scale of the operation, gives a bank something to act on, and turns
forty weak cases into one strong one.

The graph stores artefacts, not messages: a from domain, the boundary IP
the trust walk stopped at, the ASN behind it, the Reply-To address, the
DKIM selector, and every URL domain. Two messages join the same campaign
when they share enough artefact evidence.

ENOUGH is the interesting part, and it is why this module weights
artefacts by specificity instead of joining on any shared value. An ASN
is shared by millions of legitimate senders (every mail from any AWS
region shares one), so linking on ASN alone would collapse the whole
corpus into one useless campaign. A Reply-To address or a DKIM selector
is close to unique to an operation. So each artefact kind carries a
specificity weight, shared weight is summed, and a link forms only above
LINK_THRESHOLD. One high specificity artefact is enough; two low
specificity ones together are enough; one low specificity artefact on its
own is not.

Two backends behind one interface: Neo4j when the driver is installed and
configured, and an in-memory index that always works. The in-memory one
is the default, so nothing here requires a database to run.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

if __package__ in (None, ""):  # pragma: no cover - script convenience only
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

from mailguard.core.models import ParsedEmail, Verdict

# ----------------------------------------------------------------------
# Optional dependencies, guarded.
# ----------------------------------------------------------------------
try:
    from neo4j import GraphDatabase

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# How much each artefact kind is worth as evidence of a shared operation.
# These are not probabilities, they are specificity weights: how unlikely
# it is that two unrelated senders share this artefact by chance.
LINK_SPECIFICITY: dict[str, float] = {
    "reply_to": 1.0,        # an attacker-controlled mailbox, near unique
    "url_domain": 0.9,      # the landing page infrastructure
    "dkim_selector": 0.8,   # a selector string is an operator fingerprint
    "boundary_ip": 0.8,     # the last hop we can actually trust
    "from_domain": 0.6,     # unique unless it is a shared provider
    "asn": 0.3,             # shared by millions of legitimate senders
    "country": 0.05,        # recorded for context, not for linking
}

# Summed specificity needed before two messages are called one campaign.
# 0.6 means: any single high specificity artefact links, a from domain
# alone links, an ASN alone does not, and an ASN plus anything else does.
LINK_THRESHOLD: float = 0.6

# Artefact values that are worthless for linking because half the world
# shares them. Recorded on the message, never used to join campaigns.
LOW_SPECIFICITY_VALUES: frozenset[str] = frozenset({
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "icloud.com", "me.com", "aol.com", "protonmail.com",
    "proton.me", "zoho.com", "yandex.com", "mail.ru", "qq.com",
    "amazonses.com", "sendgrid.net", "mailgun.org", "mandrillapp.com",
    "sparkpostmail.com", "mailchimp.com", "salesforce.com", "zendesk.com",
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "docs.google.com",
    "drive.google.com", "sharepoint.com", "onedrive.live.com",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _artefact_key(kind: str, value: str) -> str:
    return f"{kind}:{value.strip().lower()}"


def _dkim_selectors(email: ParsedEmail, verdict: Optional[Verdict]) -> list[str]:
    """DKIM selectors, taken from x1's details or parsed from the headers.

    x1 owns DKIM verification and normally records the selector in its
    SignalResult.details; this reads several plausible key names because
    that field is the other engineer's to name. The header fallback keeps
    the graph useful when x1 has not run at all.
    """
    selectors: list[str] = []
    for signal in (verdict.signals if verdict else []) or []:
        details = signal.details or {}
        for key in ("dkim_selector", "selector", "dkim_selectors", "selectors"):
            value = details.get(key)
            if isinstance(value, str) and value.strip():
                selectors.append(value.strip())
            elif isinstance(value, (list, tuple)):
                selectors.extend(str(item).strip() for item in value if str(item).strip())
        dkim = details.get("dkim")
        if isinstance(dkim, dict):
            value = dkim.get("selector") or dkim.get("s")
            if isinstance(value, str) and value.strip():
                selectors.append(value.strip())
    if not selectors:
        # Fallback: pull the s= tag out of the raw DKIM-Signature values.
        for signature in email.dkim_signatures or []:
            for token in str(signature).split(";"):
                token = token.strip()
                if token.lower().startswith("s="):
                    candidate = token[2:].strip()
                    if candidate:
                        selectors.append(candidate)
    return sorted({s.lower() for s in selectors if s})


def extract_artefacts(
    email: ParsedEmail, verdict: Optional[Verdict] = None
) -> list[tuple[str, str]]:
    """Every artefact worth storing for this message, as (kind, value).

    Works with or without a Verdict, because find_campaign() is called
    before fusion in some flows.
    """
    artefacts: list[tuple[str, str]] = []

    if email.from_domain:
        artefacts.append(("from_domain", email.from_domain.lower()))

    reply_to = (email.reply_to or "").strip().lower()
    if reply_to:
        if "<" in reply_to and ">" in reply_to:
            reply_to = reply_to[reply_to.rfind("<") + 1 : reply_to.rfind(">")].strip()
        if reply_to:
            artefacts.append(("reply_to", reply_to))

    attribution = (verdict.attribution if verdict else None) or email.attribution
    if attribution:
        if attribution.boundary_ip:
            artefacts.append(("boundary_ip", attribution.boundary_ip))
        if attribution.asn:
            artefacts.append(("asn", str(attribution.asn)))
        if attribution.country:
            artefacts.append(("country", attribution.country))

    for selector in _dkim_selectors(email, verdict):
        artefacts.append(("dkim_selector", selector))

    for url in email.urls or []:
        domain = (url.domain or "").strip().lower()
        if domain:
            artefacts.append(("url_domain", domain))

    # De-duplicate while keeping the order stable for reproducible ids.
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for artefact in artefacts:
        if artefact not in seen:
            seen.add(artefact)
            unique.append(artefact)
    return unique


def linkable_artefacts(artefacts: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Artefacts allowed to join campaigns, dropping the worthless ones."""
    result: list[tuple[str, str]] = []
    for kind, value in artefacts:
        if LINK_SPECIFICITY.get(kind, 0.0) <= 0.0:
            continue
        if value.lower() in LOW_SPECIFICITY_VALUES:
            continue
        result.append((kind, value))
    return result


class _Neo4jBackend:
    """Write-through mirror of the graph into Neo4j, for analyst queries.

    Campaign resolution stays in the in-memory index even when this
    backend is active, for two reasons: resolution has to work when the
    database is unreachable, and the union-find logic is easier to audit
    in Python than in Cypher. Neo4j receives the messages, the artefacts,
    the edges between them and the campaign assignment, which is what an
    investigator actually wants to query interactively.
    """

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None) -> None:
        if not HAS_NEO4J:
            raise RuntimeError("neo4j driver is not installed")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def write(
        self,
        message_id: str,
        raw_sha256: str,
        campaign_id: str,
        artefacts: list[tuple[str, str]],
        probability: float,
        action: str,
        seen_at: str,
    ) -> None:
        statements = [
            (
                "MERGE (m:Message {message_id: $message_id}) "
                "SET m.raw_sha256 = $raw_sha256, m.probability = $probability, "
                "    m.action = $action, m.seen_at = $seen_at "
                "MERGE (c:Campaign {campaign_id: $campaign_id}) "
                "MERGE (m)-[:IN_CAMPAIGN]->(c)",
                {
                    "message_id": message_id,
                    "raw_sha256": raw_sha256,
                    "campaign_id": campaign_id,
                    "probability": float(probability),
                    "action": action,
                    "seen_at": seen_at,
                },
            )
        ]
        for kind, value in artefacts:
            statements.append(
                (
                    "MERGE (a:Artefact {kind: $kind, value: $value}) "
                    "WITH a MATCH (m:Message {message_id: $message_id}) "
                    "MERGE (m)-[:SHARES]->(a)",
                    {"kind": kind, "value": value, "message_id": message_id},
                )
            )
        try:
            with self._driver.session(database=self._database) as session:
                for query, parameters in statements:
                    session.run(query, **parameters)
        except Exception:
            # A graph database being down must never change a verdict.
            pass

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass


class CampaignGraph:
    """Artefact graph with campaign resolution over one or two backends."""

    def __init__(
        self,
        backend: str = "memory",
        store_path: str = "",
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        link_threshold: float = LINK_THRESHOLD,
    ) -> None:
        """Open a campaign graph.

        backend: "memory" (default, always works), "neo4j" (mirror writes
        into a database as well), or "auto" which uses Neo4j when the
        driver is installed and a URI is configured and falls back
        silently otherwise. The default is the fallback on purpose:
        nothing in MailGuard should require infrastructure to produce a
        verdict.
        """
        self.link_threshold = link_threshold
        self.store_path = store_path or os.environ.get("MAILGUARD_GRAPH_PATH", "")

        # In-memory index, the source of truth for campaign resolution.
        self._artefact_messages: dict[str, set[str]] = {}
        # One artefact can legitimately sit in more than one campaign: a
        # hosting ASN is shared, and two operations can reuse one landing
        # domain. Storing a set rather than a single id keeps the weighting
        # honest instead of letting the most recent write win.
        self._artefact_campaign: dict[str, set[str]] = {}
        self._campaigns: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, str] = {}
        self._messages: dict[str, dict[str, Any]] = {}

        # NetworkX is used only as an optional richer in-memory view; the
        # plain dict index above is what resolution reads, so the module
        # behaves identically with and without it.
        self._nx_graph = nx.Graph() if HAS_NETWORKX else None

        self.neo4j: Optional[_Neo4jBackend] = None
        uri = uri or os.environ.get("MAILGUARD_NEO4J_URI") or ""
        user = user or os.environ.get("MAILGUARD_NEO4J_USER") or "neo4j"
        password = password or os.environ.get("MAILGUARD_NEO4J_PASSWORD") or ""
        if backend in ("neo4j", "auto") and HAS_NEO4J and uri:
            try:
                self.neo4j = _Neo4jBackend(uri, user, password, database)
            except Exception:
                self.neo4j = None
        self.backend = "neo4j+memory" if self.neo4j is not None else "memory"

        if self.store_path and os.path.exists(self.store_path):
            self.load()

    # ------------------------------------------------------------------
    # Campaign bookkeeping
    # ------------------------------------------------------------------
    def _resolve(self, campaign_id: str) -> str:
        """Follow merge aliases to the surviving campaign id."""
        seen: set[str] = set()
        current = campaign_id
        while current in self._aliases and current not in seen:
            seen.add(current)
            current = self._aliases[current]
        return current

    def _new_campaign_id(self, email: ParsedEmail, artefacts: list[tuple[str, str]]) -> str:
        seed = "|".join(sorted(_artefact_key(k, v) for k, v in artefacts))
        seed += "|" + (email.message_id or email.raw_sha256 or "")
        digest = hashlib.sha256(seed.encode("utf-8", "replace")).hexdigest()[:10]
        return f"CMP-{digest.upper()}"

    def _candidate_campaigns(self, artefacts: list[tuple[str, str]]) -> dict[str, float]:
        """Campaign ids sharing artefacts with this message, with weights."""
        weights: dict[str, float] = {}
        for kind, value in linkable_artefacts(artefacts):
            key = _artefact_key(kind, value)
            for existing in self._artefact_campaign.get(key, set()):
                campaign_id = self._resolve(existing)
                if campaign_id not in self._campaigns:
                    continue
                weights[campaign_id] = weights.get(campaign_id, 0.0) + LINK_SPECIFICITY.get(kind, 0.0)
        return weights

    def _merge_campaigns(self, keep: str, drop: str) -> None:
        """Fold one campaign into another, keeping `keep` as the id."""
        if keep == drop:
            return
        kept = self._campaigns[keep]
        dropped = self._campaigns.pop(drop, None)
        if dropped is None:
            return
        kept["artefacts"].update(dropped["artefacts"])
        for message_id in dropped["messages"]:
            if message_id not in kept["messages"]:
                kept["messages"].append(message_id)
        kept["first_seen"] = min(kept["first_seen"], dropped["first_seen"])
        kept["last_seen"] = max(kept["last_seen"], dropped["last_seen"])
        kept["merged_from"] = sorted(set(kept.get("merged_from", [])) | {drop} | set(dropped.get("merged_from", [])))
        for key, campaign_ids in self._artefact_campaign.items():
            if drop in campaign_ids:
                campaign_ids.discard(drop)
                campaign_ids.add(keep)
        for record in self._messages.values():
            if record["campaign_id"] == drop:
                record["campaign_id"] = keep
        self._aliases[drop] = keep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def find_campaign(self, email: ParsedEmail, verdict: Optional[Verdict] = None) -> Optional[str]:
        """Existing campaign id for this message, or None.

        Read only: it does not create or modify anything, so it is safe to
        call before fusion to decide whether a message is part of
        something already under investigation.
        """
        artefacts = extract_artefacts(email, verdict)
        candidates = self._candidate_campaigns(artefacts)
        if not candidates:
            return None
        best_id, best_weight = max(candidates.items(), key=lambda kv: kv[1])
        return best_id if best_weight >= self.link_threshold else None

    def add_verdict(self, email: ParsedEmail, verdict: Verdict) -> str:
        """Write this message's artefacts into the graph, return a campaign id.

        If the message links to several existing campaigns, they are
        merged: the shared artefact that links them is evidence they were
        always one operation, and keeping them apart would understate the
        scale in the report.
        """
        artefacts = extract_artefacts(email, verdict)
        message_id = email.message_id or f"sha256:{email.raw_sha256}"
        seen_at = (email.date or datetime.now(timezone.utc)).isoformat(timespec="seconds") if email.date else _now_iso()

        candidates = self._candidate_campaigns(artefacts)
        linked = sorted(
            [cid for cid, weight in candidates.items() if weight >= self.link_threshold],
            key=lambda cid: (self._campaigns.get(cid, {}).get("first_seen", ""), cid),
        )

        if linked:
            campaign_id = linked[0]
            for other in linked[1:]:
                self._merge_campaigns(campaign_id, other)
        else:
            campaign_id = self._new_campaign_id(email, artefacts)
            self._campaigns[campaign_id] = {
                "campaign_id": campaign_id,
                "artefacts": set(),
                "messages": [],
                "first_seen": seen_at,
                "last_seen": seen_at,
                "merged_from": [],
            }

        campaign = self._campaigns[campaign_id]
        if message_id not in campaign["messages"]:
            campaign["messages"].append(message_id)
        campaign["first_seen"] = min(campaign["first_seen"], seen_at)
        campaign["last_seen"] = max(campaign["last_seen"], seen_at)

        for kind, value in artefacts:
            key = _artefact_key(kind, value)
            campaign["artefacts"].add(key)
            self._artefact_messages.setdefault(key, set()).add(message_id)
            if LINK_SPECIFICITY.get(kind, 0.0) > 0.0 and value.lower() not in LOW_SPECIFICITY_VALUES:
                self._artefact_campaign.setdefault(key, set()).add(campaign_id)
            if self._nx_graph is not None:
                self._nx_graph.add_node(key, kind=kind, value=value, node_type="artefact")
                self._nx_graph.add_node(message_id, node_type="message")
                self._nx_graph.add_edge(message_id, key)

        self._messages[message_id] = {
            "message_id": message_id,
            "raw_sha256": email.raw_sha256,
            "campaign_id": campaign_id,
            "probability": float(verdict.probability),
            "action": verdict.action,
            "verdict_class": verdict.verdict_class,
            "seen_at": seen_at,
            "artefacts": [_artefact_key(k, v) for k, v in artefacts],
        }

        if self.neo4j is not None:
            self.neo4j.write(
                message_id=message_id,
                raw_sha256=email.raw_sha256,
                campaign_id=campaign_id,
                artefacts=artefacts,
                probability=verdict.probability,
                action=verdict.action,
                seen_at=seen_at,
            )
        if self.store_path:
            self.save()

        verdict.campaign_id = campaign_id
        return campaign_id

    def campaign_summary(self, campaign_id: str) -> dict[str, Any]:
        """Message count, shared artefacts, and the first and last sighting."""
        resolved = self._resolve(campaign_id)
        campaign = self._campaigns.get(resolved)
        if campaign is None:
            return {
                "campaign_id": campaign_id,
                "known": False,
                "message_count": 0,
                "artefacts": {},
                "shared_artefacts": {},
                "first_seen": None,
                "last_seen": None,
            }
        messages = campaign["messages"]
        artefacts: dict[str, list[str]] = {}
        shared: dict[str, list[str]] = {}
        for key in sorted(campaign["artefacts"]):
            kind, _, value = key.partition(":")
            artefacts.setdefault(kind, []).append(value)
            # "Shared" means the artefact appears on more than one message
            # in the campaign, which is what an investigator is looking at.
            if len(self._artefact_messages.get(key, set())) > 1:
                shared.setdefault(kind, []).append(value)
        return {
            "campaign_id": resolved,
            "known": True,
            "message_count": len(messages),
            "messages": list(messages),
            "artefacts": artefacts,
            "shared_artefacts": shared,
            "first_seen": campaign["first_seen"],
            "last_seen": campaign["last_seen"],
            "merged_from": campaign.get("merged_from", []),
            "backend": self.backend,
        }

    def messages_for_artefact(self, kind: str, value: str) -> list[str]:
        """Every message that carried one artefact."""
        return sorted(self._artefact_messages.get(_artefact_key(kind, value), set()))

    def stats(self) -> dict[str, Any]:
        """Counts for a status line or a test."""
        return {
            "backend": self.backend,
            "campaigns": len(self._campaigns),
            "messages": len(self._messages),
            "artefacts": len(self._artefact_messages),
            "networkx": HAS_NETWORKX,
            "neo4j_driver": HAS_NEO4J,
        }

    # ------------------------------------------------------------------
    # Optional persistence. Production uses the database; this keeps a
    # single-host deployment useful across runs.
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> str:
        target = path or self.store_path
        if not target:
            return ""
        payload = {
            "campaigns": {
                cid: {
                    **{k: v for k, v in campaign.items() if k != "artefacts"},
                    "artefacts": sorted(campaign["artefacts"]),
                }
                for cid, campaign in self._campaigns.items()
            },
            "artefact_messages": {k: sorted(v) for k, v in self._artefact_messages.items()},
            "artefact_campaign": {k: sorted(v) for k, v in self._artefact_campaign.items()},
            "aliases": dict(self._aliases),
            "messages": dict(self._messages),
        }
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return target

    def load(self, path: Optional[str] = None) -> bool:
        target = path or self.store_path
        if not target or not os.path.exists(target):
            return False
        try:
            with open(target, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            return False
        self._campaigns = {
            cid: {**campaign, "artefacts": set(campaign.get("artefacts", []))}
            for cid, campaign in (payload.get("campaigns") or {}).items()
        }
        self._artefact_messages = {
            key: set(values) for key, values in (payload.get("artefact_messages") or {}).items()
        }
        self._artefact_campaign = {
            key: (set(value) if isinstance(value, (list, set, tuple)) else {str(value)})
            for key, value in (payload.get("artefact_campaign") or {}).items()
        }
        self._aliases = dict(payload.get("aliases") or {})
        self._messages = dict(payload.get("messages") or {})
        return True

    def close(self) -> None:
        if self.neo4j is not None:
            self.neo4j.close()


if __name__ == "__main__":  # pragma: no cover - demonstration
    from mailguard.core.models import Attribution, ExtractedURL, SignalResult

    def fake_email(
        message_id: str, from_domain: str, reply_to: str, asn: str, url_domain: str
    ) -> ParsedEmail:
        return ParsedEmail(
            message_id=message_id,
            raw_sha256=hashlib.sha256(message_id.encode()).hexdigest(),
            raw_bytes=b"",
            headers={},
            from_display="Accounts",
            from_address=f"accounts@{from_domain}",
            from_domain=from_domain,
            to_addresses=["finance@victim.example"],
            subject="Updated payment instructions",
            text_body="Please note our new bank account details.",
            html_body="",
            urls=[ExtractedURL(url=f"https://{url_domain}/pay", source="html", domain=url_domain)],
            attachments=[],
            inline_images=[],
            dkim_signatures=[],
            received_chain=[],
            references=[],
            reply_to=reply_to,
            attribution=Attribution(
                tier=2,
                tier_name="Infrastructure attributed",
                boundary_ip=None,
                asn=asn,
                country="NL",
                isp="Example Hosting",
            ),
        )

    def fake_verdict(email: ParsedEmail, dkim_selector: str) -> Verdict:
        return Verdict(
            probability=0.93,
            verdict_class="Fraud",
            action="BLOCK",
            contributions={"x2": 3.2, "x5": 2.7, "_intercept": -4.2},
            signals=[
                SignalResult(
                    signal_id="x1",
                    name="Authentication",
                    score=0.0,
                    status="ok",
                    evidence_row="SPF pass, DKIM pass, DMARC aligned",
                    details={"dkim_selector": dkim_selector},
                )
            ],
            attribution=email.attribution,
        )

    graph = CampaignGraph(backend="memory")
    print("MailGuard campaign graph demonstration")
    print(f"backend: {graph.backend}  (networkx installed: {HAS_NETWORKX}, "
          f"neo4j driver installed: {HAS_NEO4J})")
    print("-" * 78)

    # Two mails from DIFFERENT sending domains, with DIFFERENT landing
    # pages and DIFFERENT DKIM selectors, sharing exactly two things: one
    # ASN and one Reply-To address.
    first = fake_email(
        "<a-1@hdfc-verify.example>",
        "hdfc-verify.example",
        "recovery.desk01@mailbox.example",
        "AS200000",
        "hdfc-secure-login.example",
    )
    second = fake_email(
        "<b-7@axis-alerts.example>",
        "axis-alerts.example",
        "recovery.desk01@mailbox.example",
        "AS200000",
        "axis-verify-now.example",
    )

    first_campaign = graph.add_verdict(first, fake_verdict(first, "sel-a1"))
    print(f"mail 1  from {first.from_domain:<26} -> campaign {first_campaign}")
    second_campaign = graph.add_verdict(second, fake_verdict(second, "sel-b7"))
    print(f"mail 2  from {second.from_domain:<26} -> campaign {second_campaign}")
    print()
    if first_campaign == second_campaign:
        print("LINKED: both mails landed in the same campaign, although they were sent")
        print("        from different domains, with different landing pages and")
        print("        different DKIM selectors. The shared Reply-To (1.00) and the")
        print(f"        shared ASN (0.30) sum to 1.30, over the {LINK_THRESHOLD:.2f} threshold.")
    else:
        print("NOT LINKED - this contradicts the demonstration and is a bug.")

    summary = graph.campaign_summary(first_campaign)
    print()
    print(f"campaign {summary['campaign_id']}: {summary['message_count']} messages, "
          f"first seen {summary['first_seen']}, last seen {summary['last_seen']}")
    print("shared artefacts (present on more than one message):")
    for kind, values in sorted(summary["shared_artefacts"].items()):
        weight = LINK_SPECIFICITY.get(kind, 0.0)
        for value in values:
            print(f"    {kind:<14} {value:<34} specificity {weight:.2f}")

    # A third mail sharing ONLY the ASN must NOT join: an ASN on its own
    # is worth 0.30, below the threshold. This is the check that keeps the
    # graph from collapsing every sender behind one hosting provider into
    # a single meaningless campaign.
    third = fake_email(
        "<c-3@unrelated-shop.example>",
        "unrelated-shop.example",
        "orders@unrelated-shop.example",
        "AS200000",
        "unrelated-shop.example",
    )
    third_campaign = graph.add_verdict(third, fake_verdict(third, "sel-c3"))
    print()
    print(f"mail 3  shares only the ASN         -> campaign {third_campaign}")
    if third_campaign != first_campaign:
        print(
            "CORRECT: a separate campaign, because an ASN alone scores 0.30, under "
            f"the {LINK_THRESHOLD:.2f} threshold."
        )
    else:
        print("WRONG: an ASN alone should not link two senders. This is a bug.")
    print()
    print("graph stats:", graph.stats())
