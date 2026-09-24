"""Configuration for the forensics half of MailGuard AI.

The trust boundary walk is only as good as the list of hosts it is told we
control. That list is operator knowledge, not something a message can tell
us, so it lives here with the other operator settings: which provider
egress hosts, hosting ASNs and anonymising networks to recognise, where an
optional GeoLite2 database lives, and how long a signal may run.

Every value has a sensible default so the CLI runs out of the box. Every
value can be overridden by an environment variable through load_config(),
so a deployment never has to edit this file.

Nothing here touches the network or the disk at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

# Mail servers WE control, matched as a case insensitive suffix against the
# `by` host of each Received hop. REPLACE THESE with your own MX, filtering
# gateways and mailbox servers, plus any relay you deliberately route
# through. A hop is only believed if one of these wrote it.
TRUSTED_HOSTS: list[str] = ["mx.example.org", "gw.example.org"]

# Hostname fragments of large webmail providers and email service providers.
# When the trust boundary sits on one of these, attribution can name the
# provider but not the person behind the account.
PROVIDER_EGRESS_DOMAINS: dict[str, str] = {
    "google.com": "Google",
    "googlemail.com": "Google",
    "gmail": "Google",
    "outlook": "Microsoft",
    "hotmail": "Microsoft",
    "protection.outlook": "Microsoft",
    "yahoo": "Yahoo",
    "yahoodns": "Yahoo",
    "zoho": "Zoho",
    "sendgrid": "SendGrid",
    "mailgun": "Mailgun",
    "amazonses": "Amazon SES",
    "mailchimp": "Mailchimp",
    "mcsv.net": "Mailchimp",
    "mandrillapp": "Mailchimp",
    "sparkpostmail": "SparkPost",
    "postmarkapp": "Postmark",
    "sendinblue": "Brevo",
    "icloud": "Apple",
    "protonmail": "Proton",
}

# ASN / ISP name keywords that mark hosting rather than a residential or
# business access network.
DATACENTRE_ASN_KEYWORDS: list[str] = [
    "amazon", "aws", "google", "digitalocean", "ovh", "hetzner", "linode",
    "akamai", "vultr", "choopa", "azure", "microsoft", "cloudflare",
    "contabo", "leaseweb", "scaleway", "alibaba", "tencent", "oracle",
    "hosting", "datacenter", "data center", "server", "vps",
]

# Keywords that mark an anonymising layer. Matched against the boundary
# host name and the ASN organisation name.
VPN_TOR_KEYWORDS: list[str] = [
    "nordvpn", "expressvpn", "protonvpn", "mullvad", "tor exit", "torexit",
    "tor-exit", "privateinternetaccess", "private internet access",
    "surfshark", "cyberghost", "ipvanish", "windscribe", "hide.me",
    "privado", "purevpn", "anonymizer", "vpn", "proxy",
]

# Optional. Path to a MaxMind GeoLite2-City or GeoLite2-Country .mmdb (and
# optionally GEOLITE_ASN_DB_PATH for GeoLite2-ASN). When empty, or when the
# geoip2 library is not installed, attribution falls back cleanly to a stub
# that returns no location and says so.
GEOLITE_DB_PATH: str = ""
GEOLITE_ASN_DB_PATH: str = ""

# Optional. AbuseIPDB API key for ASN / IP reputation in x3. Empty skips
# that sub check.
ABUSEIPDB_KEY: str = ""

# Per signal time budget. A signal that overruns is recorded as abstain.
SIGNAL_TIMEOUT_MS: int = 400

# Live DNS / RDAP / WHOIS lookups inside signals. Set MAILGUARD_NETWORK=0
# for fully offline, reproducible analysis of stored evidence.
NETWORK_LOOKUPS: bool = True

# Timeout for any single network request made by a signal, in seconds.
NETWORK_TIMEOUT_S: float = 2.0


@dataclass
class Config:
    """All settings in one object, as returned by load_config()."""

    trusted_hosts: list[str] = field(default_factory=lambda: list(TRUSTED_HOSTS))
    provider_egress_domains: dict[str, str] = field(default_factory=lambda: dict(PROVIDER_EGRESS_DOMAINS))
    datacentre_asn_keywords: list[str] = field(default_factory=lambda: list(DATACENTRE_ASN_KEYWORDS))
    vpn_tor_keywords: list[str] = field(default_factory=lambda: list(VPN_TOR_KEYWORDS))
    geolite_db_path: str = GEOLITE_DB_PATH
    geolite_asn_db_path: str = GEOLITE_ASN_DB_PATH
    abuseipdb_key: str = ABUSEIPDB_KEY
    signal_timeout_ms: int = SIGNAL_TIMEOUT_MS
    network_lookups: bool = NETWORK_LOOKUPS
    network_timeout_s: float = NETWORK_TIMEOUT_S

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, without secrets, for printing and JSON output."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["abuseipdb_key"] = "set" if self.abuseipdb_key else ""
        return data


def _env_list(name: str) -> Optional[list[str]]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str) -> Optional[int]:
    try:
        value = os.environ.get(name)
        return int(value) if value and value.strip() else None
    except ValueError:
        return None


def _env_float(name: str) -> Optional[float]:
    try:
        value = os.environ.get(name)
        return float(value) if value and value.strip() else None
    except ValueError:
        return None


def load_config(**overrides: Any) -> Config:
    """Build a Config from the defaults above, environment variables, and overrides.

    Environment variables:
        MAILGUARD_TRUSTED_HOSTS          comma separated, replaces the default list
        MAILGUARD_PROVIDER_EGRESS        comma separated fragment=Label pairs, added
        MAILGUARD_DATACENTRE_KEYWORDS    comma separated, added
        MAILGUARD_VPN_TOR_KEYWORDS       comma separated, added
        MAILGUARD_GEOLITE_DB             path to a GeoLite2 City / Country .mmdb
        MAILGUARD_GEOLITE_ASN_DB         path to a GeoLite2 ASN .mmdb
        MAILGUARD_ABUSEIPDB_KEY          AbuseIPDB API key
        MAILGUARD_SIGNAL_TIMEOUT_MS      integer milliseconds
        MAILGUARD_NETWORK                0 to disable live lookups
        MAILGUARD_NETWORK_TIMEOUT        seconds, float

    Keyword arguments named after Config fields win over everything else;
    None values are ignored so a CLI can pass unset options straight through.
    """
    config = Config()

    hosts = _env_list("MAILGUARD_TRUSTED_HOSTS")
    if hosts:
        config.trusted_hosts = hosts
    for pair in _env_list("MAILGUARD_PROVIDER_EGRESS") or []:
        if "=" in pair:
            fragment, _, label = pair.partition("=")
            config.provider_egress_domains[fragment.strip().lower()] = label.strip()
    config.datacentre_asn_keywords += [k.lower() for k in _env_list("MAILGUARD_DATACENTRE_KEYWORDS") or []]
    config.vpn_tor_keywords += [k.lower() for k in _env_list("MAILGUARD_VPN_TOR_KEYWORDS") or []]
    config.geolite_db_path = os.environ.get("MAILGUARD_GEOLITE_DB", config.geolite_db_path)
    config.geolite_asn_db_path = os.environ.get("MAILGUARD_GEOLITE_ASN_DB", config.geolite_asn_db_path)
    config.abuseipdb_key = os.environ.get("MAILGUARD_ABUSEIPDB_KEY", config.abuseipdb_key)
    timeout = _env_int("MAILGUARD_SIGNAL_TIMEOUT_MS")
    if timeout is not None and timeout > 0:
        config.signal_timeout_ms = timeout
    network = _env_bool("MAILGUARD_NETWORK")
    if network is not None:
        config.network_lookups = network
    net_timeout = _env_float("MAILGUARD_NETWORK_TIMEOUT")
    if net_timeout is not None and net_timeout > 0:
        config.network_timeout_s = net_timeout

    known = {f.name for f in fields(Config)}
    for key, value in overrides.items():
        if key in known and value is not None:
            setattr(config, key, value)
    return config


def host_matches(host: Optional[str], suffixes: list[str]) -> Optional[str]:
    """Return the entry a host matches as a case insensitive suffix, or None.

    `mx.example.org` matches the entry `mx.example.org` and the entry
    `example.org`, but `evilexample.org` does not match `example.org`:
    a suffix must start at a label boundary, otherwise any attacker could
    register a name that ends with ours.
    """
    if not host:
        return None
    value = host.strip().strip("[]").rstrip(".").lower()
    for entry in suffixes:
        suffix = entry.strip().rstrip(".").lower().lstrip("*").lstrip(".")
        if suffix and (value == suffix or value.endswith("." + suffix)):
            return entry
    return None


# The config signals read. Signals have the fixed signature run(email), so
# the CLI sets this once per run; a signal called on its own gets one built
# from the environment on first use.
_ACTIVE: Optional[Config] = None


def set_active_config(config: Optional[Config]) -> None:
    """Make `config` the one every signal reads. None resets to defaults."""
    global _ACTIVE
    _ACTIVE = config


def get_config() -> Config:
    """The config set by the CLI, or one built from the environment."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_config()
    return _ACTIVE
