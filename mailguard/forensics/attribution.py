"""Three Tier Origin Attribution: how far can the sender of this message honestly be traced?

Attribution starts from exactly one fact: the boundary IP, the address our
own server recorded when the message crossed into our infrastructure (see
trust_boundary). Everything below the boundary is a claim and is only ever
reported as one.

  tier 1  direct_ip         the boundary IP belongs to the sender's own network:
                            not a known provider, not hosting, not anonymised.
                            ASN, country and ISP describe the origin.
  tier 2  provider_bounded  the boundary is a webmail provider, an email service
                            provider or a hosting company. The provider is
                            established; the person behind the account is
                            not, and only the provider can say who it is.
  tier 3  anonymised        the boundary sits on a VPN, a proxy or Tor, or the
                            chain could not be walked at all. No sender
                            location can be stated.

A tool claiming it can always find the attacker's real location is
overselling. Most fraud arrives through a webmail account, a rented server
or an anonymising network, and the honest answer then stops at that
infrastructure. Tier 3 therefore reports the concealment itself as the
finding: ordinary correspondence does not arrive through an anonymising
layer, and the effort taken to hide the origin is evidence about intent.

IP facts come from MaxMind GeoLite2 through the optional `geoip2` library
when GEOLITE_DB_PATH (and optionally GEOLITE_ASN_DB_PATH) is configured.
Without them classify_ip() returns a stub with no location and says so.
A City database also yields region, city and coordinates; coordinates are
kept only for tier 1, so the trace map can never pin a provider or an
anonymiser as the sender. Tor exits are recognised from a local copy of
the Tor Project exit list (TOR_EXIT_LIST_PATH) as well as by keyword.
No network call is ever made by default.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from mailguard.core.config import Config, get_config
from mailguard.core.models import Attribution, ParsedEmail
from mailguard.forensics.trust_boundary import boundary_helo, boundary_host, boundary_ip, is_internal_ip

try:
    import geoip2.database
    import geoip2.errors

    HAS_GEOIP2 = True
except ImportError:
    HAS_GEOIP2 = False

TIER_NAMES: dict[int, str] = {1: "direct_ip", 2: "provider_bounded", 3: "anonymised"}

# path -> (mtime, set of exit IPs), so a batch run reads the list once.
_TOR_CACHE: dict[str, tuple[float, frozenset[str]]] = {}


def tor_exits(path: str) -> frozenset[str]:
    """Exit IPs from a local Tor exit list (one per line, # comments). Empty when unset."""
    if not path or not os.path.exists(path):
        return frozenset()
    try:
        mtime = os.path.getmtime(path)
        cached = _TOR_CACHE.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            ips = frozenset(
                line.split()[0].strip() for line in fh if line.strip() and not line.lstrip().startswith("#")
            )
        _TOR_CACHE[path] = (mtime, ips)
        return ips
    except OSError:
        return frozenset()


def _matches(text: Optional[str], keywords: list[str]) -> Optional[str]:
    """First keyword found in text (case insensitive substring), or None."""
    if not text:
        return None
    lowered = text.lower()
    for keyword in keywords:
        if keyword and keyword.lower() in lowered:
            return keyword
    return None


def _provider_for(host: Optional[str], providers: dict[str, str]) -> Optional[str]:
    """Provider label when the host contains a known provider fragment."""
    if not host:
        return None
    lowered = host.lower()
    for fragment, label in providers.items():
        if fragment.lower() in lowered:
            return label
    return None


def classify_ip(ip: str, config: Optional[Config] = None) -> dict[str, Any]:
    """ASN, country, ISP and network type for one IP address. Never raises.

    Returns keys: ip, asn, country, region, city, latitude, longitude, isp,
    is_datacentre, is_vpn_or_tor, is_tor_exit, internal,
    lookup_unavailable, source.
    """
    config = config or get_config()
    info: dict[str, Any] = {
        "ip": ip, "asn": None, "country": None, "region": None, "city": None,
        "latitude": None, "longitude": None, "isp": None,
        "is_datacentre": False, "is_vpn_or_tor": False, "is_tor_exit": False,
        "internal": is_internal_ip(ip), "lookup_unavailable": True, "source": None,
    }
    if not ip or info["internal"]:
        return info

    if HAS_GEOIP2:
        city_path = config.geolite_db_path
        asn_path = config.geolite_asn_db_path
        if city_path and os.path.exists(city_path):
            try:
                with geoip2.database.Reader(city_path) as reader:
                    try:
                        record = reader.city(ip)
                    except Exception:
                        record = reader.country(ip)
                    if record.country and record.country.iso_code:
                        info["country"] = record.country.iso_code
                        info["lookup_unavailable"] = False
                        info["source"] = "geolite2"
                    # City-level fields exist only on a City database record.
                    subdivisions = getattr(record, "subdivisions", None)
                    if subdivisions and subdivisions.most_specific.name:
                        info["region"] = subdivisions.most_specific.name
                    city = getattr(record, "city", None)
                    if city is not None and city.name:
                        info["city"] = city.name
                    location = getattr(record, "location", None)
                    if location is not None and location.latitude is not None:
                        info["latitude"] = float(location.latitude)
                        info["longitude"] = float(location.longitude)
            except Exception:
                pass
        if asn_path and os.path.exists(asn_path):
            try:
                with geoip2.database.Reader(asn_path) as reader:
                    record = reader.asn(ip)
                    if record.autonomous_system_number:
                        info["asn"] = f"AS{record.autonomous_system_number}"
                    info["isp"] = record.autonomous_system_organization
                    info["lookup_unavailable"] = False
                    info["source"] = "geolite2"
            except Exception:
                pass

    info["is_datacentre"] = bool(_matches(info["isp"], config.datacentre_asn_keywords))
    info["is_tor_exit"] = ip in tor_exits(config.tor_exit_list_path)
    info["is_vpn_or_tor"] = info["is_tor_exit"] or bool(_matches(info["isp"], config.vpn_tor_keywords))
    return info


def _claims_below(email: ParsedEmail) -> str:
    """Printable list of what the attacker writable hops claim."""
    index = email.trust_boundary_index
    chain = email.received_chain or []
    if index is None:
        return ""
    claims = [f"hop {hop.index} claims {hop.from_ip}" for hop in chain[index:] if hop.from_ip]
    if not claims:
        return ""
    return "; unverifiable claims below the boundary: " + ", ".join(claims[:4])


def attribute(email: ParsedEmail, config: Optional[Config] = None) -> Attribution:
    """Attribute a message whose trust boundary has already been annotated. Never raises."""
    config = config or get_config()
    try:
        chain = email.received_chain or []
        index = email.trust_boundary_index

        # Internal mail: every hop is ours and the bottom one came from
        # inside. There is no outside origin to hide, so this is tier 1.
        if chain and index is None:
            origin_ip = chain[-1].from_ip
            return Attribution(
                tier=1,
                tier_name=TIER_NAMES[1],
                boundary_ip=origin_ip,
                notes="every hop was written by our own servers; the message originated inside our infrastructure",
            )

        ip = boundary_ip(chain, index)
        host = boundary_host(chain, index)
        if not ip:
            reason = (
                "there are no Received headers"
                if not chain
                else "not even the top hop was written by our servers"
                if index == 0
                else "our last trusted hop recorded no peer IP"
            )
            return Attribution(
                tier=3,
                tier_name=TIER_NAMES[3],
                notes=f"the chain could not be walked: {reason}, so no origin can be stated{_claims_below(email)}",
            )

        info = classify_ip(ip, config)
        email.meta["attribution_lookup"] = info
        common: dict[str, Any] = {
            "boundary_ip": ip,
            "asn": info["asn"],
            "country": info["country"],
            "region": info["region"],
            "city": info["city"],
            "isp": info["isp"],
        }
        lookup_note = "; IP intelligence unavailable (no GeoLite2 database)" if info["lookup_unavailable"] else ""
        helo = boundary_helo(chain, index)
        if not host and helo:
            # No reverse DNS: name the claimed HELO, but never match on it.
            lookup_note = f"; no reverse DNS, the host called itself '{helo}' (unverified)" + lookup_note

        anonymiser = ("Tor exit list" if info["is_tor_exit"] else None) or _matches(host, config.vpn_tor_keywords) or (
            _matches(info["isp"], config.vpn_tor_keywords) if info["isp"] else None
        )
        if anonymiser or info["is_vpn_or_tor"]:
            return Attribution(
                tier=3,
                tier_name=TIER_NAMES[3],
                is_vpn_or_tor=True,
                notes=(
                    f"boundary {ip} ({host or 'no reverse DNS'}) sits on anonymising infrastructure "
                    f"(matched '{anonymiser or 'ASN keyword'}'); the origin was deliberately concealed, "
                    f"and the concealment is itself the finding{_claims_below(email)}"
                ),
                **common,
            )

        provider = _provider_for(host, config.provider_egress_domains)
        datacentre = _matches(info["isp"], config.datacentre_asn_keywords) if info["isp"] else None
        if provider or datacentre:
            label = provider or f"hosting network ({info['isp']})"
            return Attribution(
                tier=2,
                tier_name=TIER_NAMES[2],
                notes=(
                    f"boundary {ip} ({host or 'no reverse DNS'}) belongs to {label}; attribution stops at "
                    f"the provider, which alone can identify the account holder{lookup_note}{_claims_below(email)}"
                ),
                **common,
            )

        # Coordinates only at tier 1. At tier 2 and 3 a point on a map would
        # be the provider's or the anonymiser's location presented as the
        # sender's, which is the false pin the trace map must never draw.
        return Attribution(
            tier=1,
            tier_name=TIER_NAMES[1],
            notes=(
                f"boundary {ip} ({host or 'no reverse DNS'}) connected directly to our server and is not a "
                f"known provider, hosting network or anonymiser{lookup_note}{_claims_below(email)}"
            ),
            latitude=info["latitude"],
            longitude=info["longitude"],
            **common,
        )
    except Exception as exc:
        return Attribution(tier=3, tier_name=TIER_NAMES[3], notes=f"attribution failed: {type(exc).__name__}")
