"""The trace map: where a message came from, drawn without a false pin.

A map invites the reader to believe every point on it, so this one is
strict about what it draws:

  tier 1   one pin, at the boundary IP's GeoLite2 City location, labelled
           as city level. This is the sender's own network.
  tier 2   no pin. The boundary is a provider, ESP or hosting network, and
           a pin would put the provider's datacentre on the map as if it
           were the sender. The map says where attribution stopped and who
           can take it further.
  tier 3   no pin. The origin was concealed (VPN, proxy, Tor) or the chain
           could not be walked. The concealment is the finding.

Hops below the trust boundary are listed as claims and never plotted: an
attacker can write any IP they like into those headers, and a map of a
fabricated route is worse than no map.

Output is one self-contained HTML file. Leaflet and the OpenStreetMap
tiles are fetched by the analyst's browser when the file is opened;
nothing here makes a network call.
"""
from __future__ import annotations

import html
import json
import os
from typing import Any, Optional

from mailguard.core.models import ParsedEmail, Verdict
from mailguard.evidence.ledger import case_id_for

LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"

TIER_EXPLANATION: dict[int, str] = {
    1: "Direct origin: the sender's own network connected to our server. Pinned at city level.",
    2: "Provider bounded: attribution stops at the provider, which alone can identify the "
       "account holder. No pin is drawn, because the provider's location is not the sender's.",
    3: "Anonymised: the origin was concealed or the chain could not be walked. No location "
       "can be stated, and the concealment is itself the finding.",
}


def pin_for(email: ParsedEmail) -> Optional[dict[str, Any]]:
    """The one point the map may draw, or None. Tier 1 with coordinates only."""
    attribution = email.attribution
    if attribution is None or attribution.tier != 1:
        return None
    if attribution.latitude is None or attribution.longitude is None:
        return None
    label = ", ".join(x for x in (attribution.city, attribution.region, attribution.country) if x)
    return {
        "lat": attribution.latitude,
        "lon": attribution.longitude,
        "label": f"{attribution.boundary_ip} ({label or 'location'}), city level",
    }


def build_trace_map(email: ParsedEmail, verdict: Optional[Verdict], out_path: str) -> str:
    """Write the trace map HTML and return its path."""
    attribution = email.attribution
    tier = attribution.tier if attribution else 3
    case_id = str(email.meta.get("case_id") or case_id_for(email.raw_sha256))
    pin = pin_for(email)
    boundary = email.trust_boundary_index

    network = ""
    if attribution is not None:
        network = ", ".join(x for x in (attribution.asn, attribution.isp, attribution.country) if x)
    if pin is None and tier == 1:
        explanation = (TIER_EXPLANATION[1].split(" Pinned")[0]
                       + " No pin: no GeoLite2 City database was configured, so no location is known.")
    else:
        explanation = TIER_EXPLANATION.get(tier, "")

    hop_rows: list[str] = []
    for hop in email.received_chain or []:
        if boundary is not None and hop.index == boundary:
            hop_rows.append("<tr class='rule'><td colspan='4'>trust boundary: everything below "
                            "was written by hosts we do not control, and is not plotted</td></tr>")
        status = "trusted" if hop.trusted else "claim, unverified"
        hop_rows.append(
            f"<tr class='{'ok' if hop.trusted else 'claim'}'><td>{hop.index}</td>"
            f"<td>{html.escape(status)}</td><td>{html.escape(hop.by_host or '?')}</td>"
            f"<td>{html.escape(hop.from_host or '?')} [{html.escape(hop.from_ip or 'no IP')}]</td></tr>"
        )

    action = verdict.action if verdict else "none"
    probability = f"{verdict.probability:.4f}" if verdict else "n/a"
    map_block = (
        "<div id='map'></div>" if pin else
        f"<div class='nomap'>No location is plotted for this message.<br>{html.escape(explanation)}</div>"
    )
    script = ""
    if pin:
        script = (
            f"<script src='{LEAFLET_JS}'></script><script>"
            f"var p={json.dumps(pin)};"
            "var m=L.map('map').setView([p.lat,p.lon],6);"
            "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',"
            "{maxZoom:12,attribution:'&copy; OpenStreetMap contributors'}).addTo(m);"
            "L.circle([p.lat,p.lon],{radius:25000}).addTo(m).bindPopup(p.label).openPopup();"
            "</script>"
        )

    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Trace map {html.escape(case_id)}</title>"
        + (f"<link rel='stylesheet' href='{LEAFLET_CSS}'>" if pin else "")
        + "<style>"
        ":root{--bg:#fff;--fg:#111;--muted:#555;--line:#ddd;--bad:#b00020;--good:#1b7a3d;--panel:#f6f6f6}"
        "@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#aaa;--line:#333;"
        "--bad:#ff6b81;--good:#5fd38d;--panel:#1e1e1e}}"
        "body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;"
        "max-width:980px;margin:0 auto;padding:16px}"
        "h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);font-size:12px}"
        "#map{height:420px;border:1px solid var(--line);margin:12px 0}"
        ".nomap{background:var(--panel);border:1px solid var(--line);padding:24px;margin:12px 0}"
        "table{border-collapse:collapse;width:100%;font-size:12.5px}td{border-bottom:1px solid var(--line);"
        "padding:4px 6px;vertical-align:top;word-break:break-all}tr.ok td:nth-child(2){color:var(--good)}"
        "tr.claim td:nth-child(2){color:var(--bad)}tr.rule td{color:var(--bad);font-weight:600;"
        "border-top:2px solid var(--bad)}</style></head><body>"
        f"<h1>Trace map</h1><div class='sub'>case {html.escape(case_id)} &middot; "
        f"verdict {html.escape(action)} (p={probability}) &middot; tier {tier} "
        f"{html.escape(attribution.tier_name if attribution else 'anonymised')}</div>"
        f"<p>Boundary IP <b>{html.escape((attribution.boundary_ip if attribution else None) or 'none')}</b>"
        + (f" &middot; {html.escape(network)}" if network else "")
        + f"</p><p class='sub'>{html.escape(explanation)}</p>"
        + map_block
        + "<h2 style='font-size:15px'>Relay path</h2><table>" + "".join(hop_rows) + "</table>"
        + script + "</body></html>"
    )
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)
    return out_path
