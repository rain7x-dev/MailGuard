"""x6 Origin and Geo: what the traceable origin says about the sender.

This signal reads `email.attribution`, which forensics.attribution has
already populated from the trust boundary IP. It never looks at an address
claimed below the boundary. Three findings:

  attribution tier   tier 3 is itself suspicious: the sender deliberately
                     concealed the origin (VPN, proxy, Tor), or the chain
                     could not be walked at all. Tier 2 is how most mail
                     arrives (a provider or ESP) and weighs little. Tier 1
                     weighs nothing.
  timezone conflict  the Date header's UTC offset is written by the sending
                     client and reflects the sender's clock. A mail claiming
                     +0530 while originating from a network in another
                     hemisphere is worth flagging: either the sender is not
                     where they present themselves, or the mail was relayed
                     through infrastructure far from them.
  datacentre origin  the boundary ASN is hosting rather than a residential or
                     business ISP, and not a known mail provider: a rented
                     server talking straight to our MX.
  forged origin      a hop below the trust boundary claims to have been written
                     by one of our own servers. Genuine mail cannot produce
                     that; a sender fabricating a travel history can. It is a
                     lie about where the message came from, so it scores here.

There is NO trained model here; this signal is rule based by design. The
rules are few, each one is a statement an analyst can check by reading the
headers, and a learned model would add opacity without adding information.

When there is no attribution at all the signal abstains.
"""
from __future__ import annotations

from typing import Any, Optional

from mailguard.core.config import get_config
from mailguard.core.models import ParsedEmail, SignalResult

SIGNAL_ID = "x6"
SIGNAL_NAME = "Origin and Geo"

TIER_RISK: dict[int, float] = {1: 0.0, 2: 0.15, 3: 0.65}
UNWALKABLE_RISK: float = 0.40       # tier 3 because the chain could not be walked, not proven concealment
TIMEZONE_CONFLICT_RISK: float = 0.30
DATACENTRE_RISK: float = 0.30
FORGED_ORIGIN_RISK: float = 0.60

# Plausible UTC offsets (minutes) for a sender in each country, standard and
# daylight time. Deliberately generous; only countries listed are checked.
COUNTRY_OFFSETS: dict[str, set[int]] = {
    "IN": {330}, "LK": {330}, "PK": {300}, "BD": {360}, "NP": {345},
    "CN": {480}, "HK": {480}, "SG": {480}, "MY": {480}, "PH": {480}, "TW": {480},
    "JP": {540}, "KR": {540}, "ID": {420, 480, 540}, "TH": {420}, "VN": {420},
    "AU": {480, 525, 570, 600, 630, 660}, "NZ": {720, 780},
    "AE": {240}, "SA": {180}, "IL": {120, 180}, "TR": {180}, "IR": {210, 270},
    "RU": {120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720},
    "UA": {120, 180}, "GB": {0, 60}, "IE": {0, 60}, "PT": {0, 60},
    "DE": {60, 120}, "FR": {60, 120}, "NL": {60, 120}, "BE": {60, 120}, "ES": {0, 60, 120},
    "IT": {60, 120}, "SE": {60, 120}, "NO": {60, 120}, "DK": {60, 120}, "FI": {120, 180},
    "PL": {60, 120}, "CH": {60, 120}, "AT": {60, 120}, "CZ": {60, 120}, "RO": {120, 180},
    "NG": {60}, "ZA": {120}, "EG": {120, 180}, "KE": {180},
    "US": {-600, -540, -480, -420, -360, -300, -240},
    "CA": {-480, -420, -360, -300, -240, -210, -180, -150},
    "MX": {-480, -420, -360, -300}, "BR": {-300, -240, -180, -120},
    "AR": {-180}, "CL": {-240, -180}, "CO": {-300},
}


def _declared_offset_minutes(email: ParsedEmail) -> Optional[int]:
    """UTC offset declared in the Date header, in minutes, or None."""
    if email.date is None or email.date.utcoffset() is None:
        return None
    raw = str(email.meta.get("date_raw") or "")
    # "-0000" means "offset unknown" (RFC 5322 section 3.3); do not read it as UTC.
    if raw.rstrip().endswith("-0000"):
        return None
    return int(email.date.utcoffset().total_seconds() // 60)


def _format_offset(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    minutes = abs(minutes)
    return f"{sign}{minutes // 60:02d}{minutes % 60:02d}"


def run(email: ParsedEmail) -> SignalResult:
    """Score the traceable origin. Never raises."""
    try:
        attribution = email.attribution
        if attribution is None:
            return SignalResult(SIGNAL_ID, SIGNAL_NAME, 0.0, "abstain",
                                "no attribution available, origin not assessed",
                                details={"abstain_reason": "attribution not populated"})

        config = get_config()
        lookup: dict[str, Any] = email.meta.get("attribution_lookup") or {}
        risks: list[tuple[float, str]] = []

        unwalkable = attribution.tier == 3 and not attribution.is_vpn_or_tor
        tier_risk = UNWALKABLE_RISK if unwalkable else TIER_RISK.get(attribution.tier, 0.0)
        tier_label = {
            1: "direct origin",
            2: "origin bounded by a provider",
            3: "chain could not be walked" if unwalkable else "origin deliberately concealed",
        }.get(attribution.tier, f"tier {attribution.tier}")
        if tier_risk > 0:
            risks.append((tier_risk, tier_label))

        offset = _declared_offset_minutes(email)
        country = (attribution.country or "").upper() or None
        conflict = False
        if offset is not None and country in COUNTRY_OFFSETS and attribution.tier != 3:
            if offset not in COUNTRY_OFFSETS[country]:
                conflict = True
                risks.append((TIMEZONE_CONFLICT_RISK, f"Date header {_format_offset(offset)} conflicts with origin country {country}"))

        isp = attribution.isp or ""
        datacentre = bool(lookup.get("is_datacentre")) or any(k in isp.lower() for k in config.datacentre_asn_keywords if k)
        provider_bounded = attribution.tier == 2 and "hosting network" not in (attribution.notes or "")
        if datacentre and not provider_bounded and attribution.tier != 3:
            risks.append((DATACENTRE_RISK, f"datacentre ASN ({isp or attribution.asn})"))

        forged = (email.meta.get("trust_boundary") or {}).get("forged_internal_hops") or []
        if forged:
            risks.append((FORGED_ORIGIN_RISK, f"hop {forged[0]} below the boundary falsely claims to be our server"))

        remaining = 1.0
        for risk, _label in risks:
            remaining *= 1.0 - risk
        score = 1.0 - remaining

        where = attribution.boundary_ip or "no boundary IP"
        network = ", ".join(x for x in (attribution.asn, attribution.isp, attribution.country) if x)
        head = f"tier {attribution.tier} {attribution.tier_name}: {where}" + (f" ({network})" if network else "")
        row = "; ".join([head] + [label for risk, label in risks if label != tier_label])
        return SignalResult(
            signal_id=SIGNAL_ID,
            name=SIGNAL_NAME,
            score=round(max(0.0, min(1.0, score)), 4),
            status="ok",
            evidence_row=row[:200],
            details={
                "tier": attribution.tier,
                "tier_name": attribution.tier_name,
                "boundary_ip": attribution.boundary_ip,
                "asn": attribution.asn,
                "country": attribution.country,
                "isp": attribution.isp,
                "is_vpn_or_tor": attribution.is_vpn_or_tor,
                "declared_utc_offset": _format_offset(offset) if offset is not None else None,
                "timezone_conflict": conflict,
                "timezone_checked": offset is not None and country in COUNTRY_OFFSETS,
                "is_datacentre": datacentre,
                "forged_internal_hops": forged,
                "lookup_unavailable": lookup.get("lookup_unavailable", True),
                "risk_reasons": [label for _risk, label in risks],
            },
        )
    except Exception as exc:  # run() must never raise
        return SignalResult(SIGNAL_ID, SIGNAL_NAME, 0.0, "abstain",
                            f"origin signal could not run: {type(exc).__name__}", details={"error": repr(exc)})
